/* (C)2026 */
package viritualisres.phonevr

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.util.Log
import android.util.Size
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.Landmark
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.sqrt
import kotlin.math.tan

/**
 * Tracks the user's hands with the phone's back camera (MediaPipe Hand Landmarker) and reports
 * them as 26 OpenXR style joints per hand, expressed in the head-local space of the headset.
 *
 * The joints are consumed by the native ALVR client (alvr_main.cpp), which converts them to
 * tracking space and sends them to the ALVR server together with the head pose. The server maps
 * them to SteamVR (skeleton + pinch gestures).
 *
 * Each joint is 7 floats: qx, qy, qz, qw, px, py, pz. Head-local axes follow OpenXR: +X right, +Y
 * up, -Z forward (the direction the user looks at).
 */
class HandTracker(private val activity: AppCompatActivity, private val listener: Listener) {

    fun interface Listener {
        /** Called for every camera frame. A null array means that the hand is not visible. */
        fun onHands(left: FloatArray?, right: FloatArray?)
    }

    private var landmarker: HandLandmarker? = null
    private var provider: ProcessCameraProvider? = null
    private var analysis: ImageAnalysis? = null
    private var executor: ExecutorService? = null

    @Volatile private var running = false
    @Volatile private var focalPx = 0f
    @Volatile private var frameWidth = 0
    @Volatile private var frameHeight = 0
    private var lastTimestampMs = 0L

    fun start() {
        if (running) return
        val appContext = activity.applicationContext

        try {
            val baseOptions = BaseOptions.builder().setModelAssetPath(MODEL_ASSET).build()
            val options =
                HandLandmarker.HandLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setRunningMode(RunningMode.LIVE_STREAM)
                    .setNumHands(2)
                    .setMinHandDetectionConfidence(0.5f)
                    .setMinHandPresenceConfidence(0.5f)
                    .setMinTrackingConfidence(0.5f)
                    .setResultListener { result, _ -> onResult(result) }
                    .setErrorListener { e -> Log.e(TAG, "MediaPipe error", e) }
                    .build()
            landmarker = HandLandmarker.createFromOptions(appContext, options)
        } catch (e: Exception) {
            // Most likely the model file is missing from the assets (see downloadHandModel).
            Log.e(TAG, "Could not create the hand landmarker, hand tracking is disabled", e)
            return
        }

        running = true
        executor = Executors.newSingleThreadExecutor()

        val providerFuture = ProcessCameraProvider.getInstance(appContext)
        providerFuture.addListener(
            {
                if (!running) return@addListener
                try {
                    val cameraProvider = providerFuture.get()
                    val resolution =
                        ResolutionSelector.Builder()
                            .setResolutionStrategy(
                                ResolutionStrategy(
                                    Size(ANALYSIS_WIDTH, ANALYSIS_HEIGHT),
                                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER,
                                )
                            )
                            .build()
                    val imageAnalysis =
                        ImageAnalysis.Builder()
                            .setResolutionSelector(resolution)
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                            .build()
                    imageAnalysis.setAnalyzer(executor!!) { image -> analyze(image) }

                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        activity,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        imageAnalysis,
                    )
                    provider = cameraProvider
                    analysis = imageAnalysis
                    Log.i(TAG, "Hand tracking camera started")
                } catch (e: Exception) {
                    Log.e(TAG, "Could not start the camera for hand tracking", e)
                }
            },
            ContextCompat.getMainExecutor(appContext),
        )
    }

    fun stop() {
        running = false
        try {
            analysis?.clearAnalyzer()
            provider?.unbindAll()
        } catch (e: Exception) {
            Log.w(TAG, "Error while stopping the camera", e)
        }
        analysis = null
        provider = null
        executor?.shutdown()
        executor = null
        try {
            landmarker?.close()
        } catch (e: Exception) {
            Log.w(TAG, "Error while closing the hand landmarker", e)
        }
        landmarker = null
        listener.onHands(null, null)
    }

    @Suppress("DEPRECATION")
    private fun analyze(image: ImageProxy) {
        try {
            val detector = landmarker
            if (detector == null || !running) return

            // Keep the analysis orientation in sync with the (landscape) display orientation.
            analysis?.targetRotation = activity.windowManager.defaultDisplay.rotation

            val rotation = image.imageInfo.rotationDegrees
            val uprightSwap = rotation % 180 != 0
            frameWidth = if (uprightSwap) image.height else image.width
            frameHeight = if (uprightSwap) image.width else image.height
            if (focalPx <= 0f) focalPx = computeFocalPx(max(image.width, image.height))

            var timestampMs = image.imageInfo.timestamp / 1_000_000L
            if (timestampMs <= lastTimestampMs) timestampMs = lastTimestampMs + 1
            lastTimestampMs = timestampMs

            val mpImage = BitmapImageBuilder(image.toBitmap()).build()
            val processing = ImageProcessingOptions.builder().setRotationDegrees(rotation).build()
            detector.detectAsync(mpImage, processing, timestampMs)
        } catch (e: Exception) {
            Log.e(TAG, "Hand tracking frame failed", e)
        } finally {
            image.close()
        }
    }

    private fun onResult(result: HandLandmarkerResult) {
        var left: FloatArray? = null
        var right: FloatArray? = null

        val hands = result.landmarks()
        for (i in hands.indices) {
            val label = result.handedness()[i][0].categoryName()
            val modelSaysRight = label == "Right"
            val isRight = if (SWAP_HANDEDNESS) !modelSaysRight else modelSaysRight

            val joints =
                HandMath.toJoints(
                    hands[i],
                    result.worldLandmarks()[i],
                    isRight,
                    frameWidth,
                    frameHeight,
                    focalPx,
                ) ?: continue

            if (isRight) {
                if (right == null) right = joints
            } else {
                if (left == null) left = joints
            }
        }
        listener.onHands(left, right)
    }

    /** Focal length of the back camera in pixels of the analysis image (long side of the sensor). */
    private fun computeFocalPx(rawLongSide: Int): Float {
        try {
            val manager = activity.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            for (id in manager.cameraIdList) {
                val c = manager.getCameraCharacteristics(id)
                if (c.get(CameraCharacteristics.LENS_FACING) != CameraCharacteristics.LENS_FACING_BACK)
                    continue
                val focalMm =
                    c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull()
                        ?: continue
                val sensor = c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE) ?: continue
                val sensorLongMm = max(sensor.width, sensor.height)
                if (sensorLongMm <= 0f) continue
                return focalMm / sensorLongMm * rawLongSide
            }
        } catch (e: Exception) {
            Log.w(TAG, "Could not read camera intrinsics, using a default field of view", e)
        }
        return (rawLongSide / 2f) / tan(Math.toRadians(DEFAULT_FOV_DEG / 2.0)).toFloat()
    }

    companion object {
        private const val TAG = "HandTracker"
        private const val MODEL_ASSET = "hand_landmarker.task"
        private const val ANALYSIS_WIDTH = 640
        private const val ANALYSIS_HEIGHT = 480
        private const val DEFAULT_FOV_DEG = 65.0

        /**
         * MediaPipe assumes mirrored (selfie) images when it guesses left/right. The back camera is
         * not mirrored, so the labels have to be swapped. Set to false if your hands are swapped.
         */
        const val SWAP_HANDEDNESS = true
    }
}

/** Geometry helpers: MediaPipe landmarks -> 26 OpenXR joints in head-local space. */
internal object HandMath {

    /** Where the back camera sits relative to the middle of the head (head-local, metres). */
    private val CAMERA_IN_HEAD = V3(0f, 0f, -0.07f)

    /**
     * Sign of the depth axis of MediaPipe's world landmarks (+1: larger z is farther from the
     * camera). Flip to -1f if fingers pointing at the camera look reversed.
     */
    private const val WORLD_Z_SIGN = 1f

    /** Multiplies the estimated distance to the hand. Raise it if your hands seem too close. */
    private const val HAND_SCALE = 1f

    /** Estimates closer than this (metres) are treated as noise and the frame is dropped. */
    private const val MIN_DEPTH_M = 0.05f

    /** Pairs of MediaPipe landmarks on the palm that are used to estimate the distance. */
    private val DEPTH_SEGMENTS =
        listOf(0 to 5, 0 to 9, 0 to 13, 0 to 17, 5 to 9, 9 to 13, 13 to 17, 5 to 17)

    class V3(val x: Float, val y: Float, val z: Float) {
        operator fun plus(o: V3) = V3(x + o.x, y + o.y, z + o.z)

        operator fun minus(o: V3) = V3(x - o.x, y - o.y, z - o.z)

        operator fun times(s: Float) = V3(x * s, y * s, z * s)

        operator fun unaryMinus() = V3(-x, -y, -z)

        fun dot(o: V3) = x * o.x + y * o.y + z * o.z

        fun cross(o: V3) = V3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x)

        fun length() = sqrt(dot(this))

        fun normalized(): V3 {
            val l = length()
            return if (l < 1e-6f) V3(0f, 0f, 0f) else V3(x / l, y / l, z / l)
        }
    }

    private fun lerp(a: V3, b: V3, t: Float) = a + (b - a) * t

    fun toJoints(
        image: List<NormalizedLandmark>,
        world: List<Landmark>,
        isRight: Boolean,
        width: Int,
        height: Int,
        focalPx: Float,
    ): FloatArray? {
        if (image.size < 21 || world.size < 21 || focalPx <= 0f || width <= 0 || height <= 0) {
            return null
        }

        // Pixel coordinates relative to the image centre (x right, y down).
        val px = FloatArray(21) { image[it].x() * width - width / 2f }
        val py = FloatArray(21) { image[it].y() * height - height / 2f }
        val metric = Array(21) { V3(world[it].x(), world[it].y(), world[it].z() * WORLD_Z_SIGN) }

        // Distance to the hand from the apparent size of the palm. Foreshortening only shrinks
        // the projected length, so the least shrunk segment (largest ratio) is the best estimate.
        var bestRatio = 0f
        var bestMidDepth = 0f // depth of that segment relative to the wrist (hand model units)
        for ((a, b) in DEPTH_SEGMENTS) {
            val pixels = hypot(px[a] - px[b], py[a] - py[b])
            val meters = (metric[a] - metric[b]).length()
            if (meters > 1e-4f && pixels / meters > bestRatio) {
                bestRatio = pixels / meters
                bestMidDepth = (metric[a].z + metric[b].z) / 2f - metric[0].z
            }
        }
        if (bestRatio < 1f) return null
        // The estimate is the distance to the middle of the best segment, the wrist is
        // `bestMidDepth` closer or farther than that.
        val depth = focalPx / bestRatio * HAND_SCALE - bestMidDepth
        if (depth < MIN_DEPTH_M) return null

        // Camera space (x right, y down, z forward): wrist from the image, the rest of the hand
        // from MediaPipe's metric 3D model of the hand.
        val wrist = V3(px[0] / focalPx * depth, py[0] / focalPx * depth, depth)
        val head =
            Array(21) {
                val c = wrist + (metric[it] - metric[0])
                // OpenXR head-local axes (x right, y up, -z forward) + camera offset.
                V3(c.x, -c.y, -c.z) + CAMERA_IN_HEAD
            }

        // OpenXR joint positions.
        val p = Array(26) { V3(0f, 0f, 0f) }
        p[1] = head[0] // wrist
        p[0] = lerp(head[0], head[9], 0.5f) // palm
        p[2] = head[1] // thumb metacarpal
        p[3] = head[2] // thumb proximal
        p[4] = head[3] // thumb distal
        p[5] = head[4] // thumb tip
        for (f in 0 until 4) {
            val mp = 5 + f * 4 // MediaPipe: MCP, PIP, DIP, TIP
            val xr = 6 + f * 5 // OpenXR: metacarpal, proximal, intermediate, distal, tip
            p[xr] = lerp(head[0], head[mp], 0.25f)
            p[xr + 1] = head[mp]
            p[xr + 2] = head[mp + 1]
            p[xr + 3] = head[mp + 2]
            p[xr + 4] = head[mp + 3]
        }

        // Direction of the back of the hand. For a right hand cross(index, pinky) points out of
        // the palm, for a left hand it points out of the back of the hand.
        val across = (head[5] - head[0]).cross(head[17] - head[0]).normalized()
        val up = if (isRight) -across else across
        val palmForward = (head[9] - head[0]).normalized()

        val out = FloatArray(26 * 7)
        for (j in 0 until 26) {
            val forward =
                when {
                    j <= 1 -> palmForward
                    j in 2..4 -> (p[j + 1] - p[j]).normalized()
                    j == 5 -> (p[5] - p[4]).normalized()
                    (j - 6) % 5 == 4 -> (p[j] - p[j - 1]).normalized()
                    else -> (p[j + 1] - p[j]).normalized()
                }
            writeJoint(out, j, p[j], forward, up)
        }
        return out
    }

    /** OpenXR joint frame: -Z points along the bone to the fingertip, +Y to the back of the hand. */
    private fun writeJoint(out: FloatArray, index: Int, position: V3, forward: V3, up: V3) {
        val z = -forward
        var x = up.cross(z)
        if (x.length() < 1e-4f) {
            // Degenerate: the bone is parallel to "up". Pick any perpendicular axis.
            x = if (kotlin.math.abs(z.x) < 0.9f) V3(1f, 0f, 0f).cross(z) else V3(0f, 1f, 0f).cross(z)
        }
        x = x.normalized()
        val y = z.cross(x)

        // Rotation matrix with columns x, y, z -> quaternion.
        val m00 = x.x
        val m10 = x.y
        val m20 = x.z
        val m01 = y.x
        val m11 = y.y
        val m21 = y.z
        val m02 = z.x
        val m12 = z.y
        val m22 = z.z

        val qx: Float
        val qy: Float
        val qz: Float
        val qw: Float
        val trace = m00 + m11 + m22
        if (trace > 0f) {
            val s = sqrt(trace + 1f) * 2f
            qw = 0.25f * s
            qx = (m21 - m12) / s
            qy = (m02 - m20) / s
            qz = (m10 - m01) / s
        } else if (m00 > m11 && m00 > m22) {
            val s = sqrt(1f + m00 - m11 - m22) * 2f
            qw = (m21 - m12) / s
            qx = 0.25f * s
            qy = (m01 + m10) / s
            qz = (m02 + m20) / s
        } else if (m11 > m22) {
            val s = sqrt(1f + m11 - m00 - m22) * 2f
            qw = (m02 - m20) / s
            qx = (m01 + m10) / s
            qy = 0.25f * s
            qz = (m12 + m21) / s
        } else {
            val s = sqrt(1f + m22 - m00 - m11) * 2f
            qw = (m10 - m01) / s
            qx = (m02 + m20) / s
            qy = (m12 + m21) / s
            qz = 0.25f * s
        }

        val o = index * 7
        out[o] = qx
        out[o + 1] = qy
        out[o + 2] = qz
        out[o + 3] = qw
        out[o + 4] = position.x
        out[o + 5] = position.y
        out[o + 6] = position.z
    }
}
