package com.pocketkitty.pocket_kitty

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Handler
import android.os.Looper
import androidx.exifinterface.media.ExifInterface
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * 口袋毛孩 · Android 原生桥接（v1.1.2 诊断版）
 *
 * 【v1.1.2 变更】
 * 1. 全管线分阶段自报家门：模型加载 / 读取图片 / AI推理 / 合成透明图，
 *    任何阶段失败都会在报错里写明阶段、异常类型和详细信息——
 *    之前的"抠图失败"两字是异常信息为空时的兜底，无法定位
 * 2. 捕获范围扩大到系统级 Error（如内存不足），不再只捕获普通 Exception
 * 3. 异常类型 + 信息拼进报错（XxxError: xxx），手机上直接可读
 */
class MainActivity : FlutterActivity() {

    companion object {
        private const val CHANNEL = "pet_segmentation/segment"

        /** 构建标识：首页底部可见，报错自动带上 */
        private const val BUILD_TAG = "v1.1.2-hq"

        /** 全量版模型（84MB） */
        private const val MODEL_FILE = "u2net.tflite"
        private const val MODEL_INPUT = 320

        /** 模型候选路径，依次尝试 */
        private val MODEL_CANDIDATES = listOf(
            MODEL_FILE,
            "flutter_assets/$MODEL_FILE",
            "flutter_assets/assets/models/$MODEL_FILE"
        )

        private const val DECODE_MAX_DIM = 1440
        private const val TILE_THRESHOLD = 900
        private const val PERCENTILE_LO = 0.02f
        private const val PERCENTILE_HI = 0.98f

        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private val mainHandler = Handler(Looper.getMainLooper())
    private val executor = Executors.newSingleThreadExecutor()
    private var interpreter: Interpreter? = null

    /** 阶段包装：任何阶段抛错都会带上阶段名 + 异常类型 + 详细信息 */
    private inline fun <T> stage(name: String, block: () -> T): T {
        try {
            return block()
        } catch (t: Throwable) {
            val detail = t.message?.takeIf { it.isNotBlank() } ?: "无更多信息"
            throw RuntimeException(
                "${name}失败 [${t.javaClass.simpleName}] $detail", t
            )
        }
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "buildVersion" -> result.success(BUILD_TAG)
                    "removeBackground" -> {
                        val path = call.argument<String>("path")
                        if (path.isNullOrBlank()) {
                            result.error("BAD_ARGS", "缺少图片路径", null)
                        } else {
                            val maskSource = (call.argument<Int>("maskSource") ?: 0)
                                .coerceIn(0, 6)
                            runSegmentation(path, maskSource, result)
                        }
                    }
                    "clickSound" -> {
                        try {
                            val tg = ToneGenerator(AudioManager.STREAM_MUSIC, 80)
                            tg.startTone(ToneGenerator.TONE_PROP_BEEP, 150)
                            mainHandler.postDelayed({ tg.release() }, 400)
                            result.success(null)
                        } catch (_: Exception) {
                            result.success(null)
                        }
                    }
                    else -> result.notImplemented()
                }
            }
    }

    private fun runSegmentation(path: String, maskSource: Int, result: MethodChannel.Result) {
        executor.execute {
            try {
                val segmenter = stage("模型加载") { obtainInterpreter() }
                val src = stage("读取图片") { decodeScaled(path, DECODE_MAX_DIM) }

                // ---------- 贴片网格 ----------
                val grid = if (max(src.width, src.height) > TILE_THRESHOLD) 2 else 1
                val tileW0 = (src.width + grid - 1) / grid
                val tileH0 = (src.height + grid - 1) / grid
                val pad = max(8, (min(tileW0, tileH0) * 0.125f).roundToInt())

                // ---------- 输出缓冲（7 路固定形状，贴片间复用） ----------
                val outputs = stage("分配推理缓冲") {
                    val map = HashMap<Int, Any>()
                    Array(segmenter.outputTensorCount) { idx ->
                        val count = segmenter.getOutputTensor(idx).shape()
                            .fold(1) { acc, d -> acc * maxOf(d, 1) }
                        ByteBuffer.allocateDirect(count * 4)
                            .order(ByteOrder.nativeOrder())
                            .also { map[idx] = it }
                    }.also { map }
                }
                val buffers = outputs.first!!
                val outIdx = maskSource.coerceIn(0, segmenter.outputTensorCount - 1)

                // ---------- 逐贴片推理 + 加权融合 ----------
                val w = src.width
                val h = src.height
                val acc = FloatArray(w * h)
                val wgt = FloatArray(w * h)

                for (j in 0 until grid) {
                    for (i in 0 until grid) {
                        val x0 = max(0, i * tileW0 - pad)
                        val y0 = max(0, j * tileH0 - pad)
                        val x1 = min(w, (i + 1) * tileW0 + pad)
                        val y1 = min(h, (j + 1) * tileH0 + pad)
                        val tw = x1 - x0
                        val th = y1 - y0

                        val tile = stage("切图($i,$j)") {
                            Bitmap.createBitmap(src, x0, y0, tw, th)
                        }
                        val mask320 = stage("AI推理($i,$j)第${outIdx + 1}路") {
                            infer(segmenter, tile, outputs, outIdx)
                        }
                        val tileAlpha = stage("遮罩放大($i,$j)") {
                            upscaleMask(mask320, tw, th)
                        }

                        for (ty in 0 until th) {
                            val wy = edgeRamp(ty, th, pad, y0 == 0, y1 == h)
                            val rowA = (y0 + ty) * w
                            val rowT = ty * tw
                            for (tx in 0 until tw) {
                                val wx = edgeRamp(tx, tw, pad, x0 == 0, x1 == w)
                                val weight = wx * wy
                                if (weight <= 0f) continue
                                val gi = rowA + x0 + tx
                                acc[gi] += tileAlpha[rowT + tx] * weight
                                wgt[gi] += weight
                            }
                        }
                        tile.recycle()
                    }
                }

                // ---------- 后处理 ----------
                val alpha8 = FloatArray(w * h)
                var filled = 0
                for (i in alpha8.indices) {
                    alpha8[i] = if (wgt[i] > 0f) acc[i] / wgt[i] else 0f
                    if (alpha8[i] > 127f) filled++
                }
                if (filled < w * h * 0.01f) {
                    src.recycle()
                    throw IllegalStateException("没有识别到明确的主体，试试更清晰、宠物占比更大的照片")
                }

                robustNormalize(alpha8)
                sCurve(alpha8)
                boxBlur3(alpha8, w, h, radius = 2)

                // ---------- 合成透明 PNG ----------
                val outFile = stage("合成透明图") { composeAndSave(src, alpha8) }
                src.recycle()

                mainHandler.post { result.success(outFile.absolutePath) }
            } catch (e: Throwable) {
                mainHandler.post {
                    result.error("SEGMENT_FAIL", "[${BUILD_TAG}] ${e.message ?: "抠图失败"}", null)
                }
            }
        }
    }

    /** 合成 alpha 并写盘（独立成函数便于阶段标注） */
    private fun composeAndSave(src: Bitmap, alpha8: FloatArray): File {
        val w = src.width
        val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)
        val out = IntArray(w * h)
        for (i in pixels.indices) {
            val a = alpha8[i].roundToInt().coerceIn(0, 255)
            val p = pixels[i]
            out[i] = if (a == 0) 0
            else Color.argb(a, Color.red(p), Color.green(p), Color.blue(p))
        }

        val resultBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        resultBmp.setPixels(out, 0, w, 0, 0, w, h)

        val dir = getExternalFilesDir(null) ?: filesDir
        val file = File(dir, "cutout_${System.currentTimeMillis()}.png")
        FileOutputStream(file).use { fos ->
            resultBmp.compress(Bitmap.CompressFormat.PNG, 100, fos)
        }
        resultBmp.recycle()
        return file
    }

    /** 单次推理：bitmap → 320² 输入 → 指定侧输出读出 320² 显著性 FloatArray */
    private fun infer(
        segmenter: Interpreter,
        bitmap: Bitmap,
        outputs: HashMap<Int, Any>,
        outIdx: Int
    ): FloatArray {
        val input = buildInput(bitmap)
        segmenter.runForMultipleInputsOutputs(arrayOf(input), outputs)
        val buf = outputs[outIdx] as ByteBuffer
        buf.rewind()
        val mask = FloatArray(MODEL_INPUT * MODEL_INPUT)
        for (i in mask.indices) mask[i] = buf.float
        buf.rewind()
        return mask
    }

    /** 320² 显著性 → tw×th 灰度 Bitmap 的红色通道（0-255） */
    private fun upscaleMask(mask: FloatArray, tw: Int, th: Int): IntArray {
        val small = Bitmap.createBitmap(MODEL_INPUT, MODEL_INPUT, Bitmap.Config.ARGB_8888)
        val sp = IntArray(MODEL_INPUT * MODEL_INPUT) { i ->
            val v = (mask[i] * 255f).toInt().coerceIn(0, 255)
            Color.rgb(v, v, v)
        }
        small.setPixels(sp, 0, MODEL_INPUT, 0, 0, MODEL_INPUT, MODEL_INPUT)
        val big = Bitmap.createScaledBitmap(small, tw, th, true)
        val bp = IntArray(tw * th)
        big.getPixels(bp, 0, tw, 0, 0, tw, th)
        big.recycle()
        small.recycle()
        return bp
    }

    /** 距贴片边的羽化权重：重叠带内余弦渐变（0→1），贴图像边界侧恒为 1 */
    private fun edgeRamp(pos: Int, len: Int, band: Int, edgeStart: Boolean, edgeEnd: Boolean): Float {
        val fromStart = if (edgeStart) 1f
        else if (pos >= band) 1f
        else {
            val t = pos.toFloat() / band
            (1 - cos(t * Math.PI)).toFloat() * 0.5f
        }
        val fromEnd = if (edgeEnd) 1f
        else if (pos < len - band) 1f
        else {
            val t = (len - 1 - pos).toFloat() / band
            (1 - cos(t * Math.PI)).toFloat() * 0.5f
        }
        return min(fromStart, fromEnd)
    }

    /** 百分位归一化：只拉伸 2%~98% 分位之间的动态范围 */
    private fun robustNormalize(data: FloatArray) {
        val hist = IntArray(256)
        for (v in data) hist[v.roundToInt().coerceIn(0, 255)]++
        val total = data.size
        val loTarget = (total * PERCENTILE_LO).roundToInt()
        val hiTarget = (total * PERCENTILE_HI).roundToInt()
        var cum = 0
        var lo = 0
        var hi = 255
        for (v in 0..255) {
            cum += hist[v]
            if (cum >= loTarget) { lo = v; break }
        }
        cum = 0
        for (v in 0..255) {
            cum += hist[v]
            if (cum >= hiTarget) { hi = v; break }
        }
        if (hi - lo < 8) { lo = 0; hi = 255 }
        val range = (hi - lo).toFloat()
        for (i in data.indices) {
            val t = ((data[i] - lo) / range).coerceIn(0f, 1f)
            data[i] = t * 255f
        }
    }

    /** S 曲线（smoothstep） */
    private fun sCurve(data: FloatArray) {
        for (i in data.indices) {
            val t = data[i] / 255f
            data[i] = t * t * (3f - 2f * t) * 255f
        }
    }

    /** 三次可分离盒模糊 ≈ 高斯羽化 */
    private fun boxBlur3(data: FloatArray, w: Int, h: Int, radius: Int) {
        val tmp = FloatArray(data.size)
        repeat(3) {
            for (y in 0 until h) {
                val row = y * w
                var sum = 0f
                var count = 0
                for (x in -radius..radius) {
                    val xi = x.coerceIn(0, w - 1)
                    sum += data[row + xi]; count++
                }
                for (x in 0 until w) {
                    tmp[row + x] = sum / count
                    val outX = (x - radius).coerceIn(0, w - 1)
                    val inX = (x + radius + 1).coerceIn(0, w - 1)
                    sum += data[row + inX] - data[row + outX]
                }
            }
            for (x in 0 until w) {
                var sum = 0f
                var count = 0
                for (y in -radius..radius) {
                    val yi = y.coerceIn(0, h - 1)
                    sum += tmp[yi * w + x]; count++
                }
                for (y in 0 until h) {
                    data[y * w + x] = sum / count
                    val outY = (y - radius).coerceIn(0, h - 1)
                    val inY = (y + radius + 1).coerceIn(0, h - 1)
                    sum += tmp[inY * w + x] - tmp[outY * w + x]
                }
            }
        }
    }

    /** 懒加载模型：优先内存映射（零拷贝，内存占用减半），回退字节流读取，只初始化一次 */
    private fun obtainInterpreter(): Interpreter {
        synchronized(this) {
            interpreter?.let { return it }

            val model = loadModelBuffer()
            val segmenter = Interpreter(model, Interpreter.Options().setNumThreads(4))

            val inShape = segmenter.getInputTensor(0).shape()
            if (inShape.any { it <= 0 }) {
                segmenter.resizeInput(0, intArrayOf(1, MODEL_INPUT, MODEL_INPUT, 3))
            }
            segmenter.allocateTensors()

            interpreter = segmenter
            return segmenter
        }
    }

    private fun loadModelBuffer(): ByteBuffer {
        val tried = StringBuilder()
        for (candidate in MODEL_CANDIDATES) {
            try {
                // 首选：mmap 零拷贝（资产未被压缩时可用，内存占用最低）
                val afd = assets.openFd(candidate)
                val mapped = afd.use {
                    FileInputStream(it.fileDescriptor).channel.map(
                        java.nio.channels.FileChannel.MapMode.READ_ONLY,
                        it.startOffset,
                        it.declaredLength
                    )
                }
                return mapped
            } catch (_: Exception) {
                // 回退：字节流读取（资产被压缩时 openFd 不可用）
                try {
                    val bytes = assets.open(candidate).use { it.readBytes() }
                    val buffer = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
                    buffer.put(bytes)
                    buffer.rewind()
                    return buffer
                } catch (_: Exception) {
                    tried.append(candidate).append("  ")
                }
            }
        }
        throw IllegalStateException(
            "模型不在 APK 里（尝试过：$tried）。确认云端打包时模型下载步骤正常。"
        )
    }

    /** 原图 → 320×320 → 归一化 FloatBuffer */
    private fun buildInput(bitmap: Bitmap): ByteBuffer {
        val scaled = Bitmap.createScaledBitmap(bitmap, MODEL_INPUT, MODEL_INPUT, true)
        val pixels = IntArray(MODEL_INPUT * MODEL_INPUT)
        scaled.getPixels(pixels, 0, MODEL_INPUT, 0, 0, MODEL_INPUT, MODEL_INPUT)
        scaled.recycle()

        val buffer = ByteBuffer
            .allocateDirect(MODEL_INPUT * MODEL_INPUT * 3 * 4)
            .order(ByteOrder.nativeOrder())

        fun norm(channel: Int, ch: Int): Float =
            (channel / 255f - MEAN[ch]) / STD[ch]

        for (p in pixels) {
            buffer.putFloat(norm(Color.red(p), 0))
            buffer.putFloat(norm(Color.green(p), 1))
            buffer.putFloat(norm(Color.blue(p), 2))
        }
        buffer.rewind()
        return buffer
    }

    /** 解码 + 按最长边降采样 + 按 EXIF 方向转正 */
    private fun decodeScaled(path: String, maxDim: Int): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, bounds)
        var sample = 1
        var side = maxOf(bounds.outWidth, bounds.outHeight)
        while (side / 2 >= maxDim) {
            sample *= 2
            side /= 2
        }
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        val bmp = BitmapFactory.decodeFile(path, opts)
            ?: throw IllegalStateException("不是有效的图片文件")

        val rotation = try {
            when (ExifInterface(path).getAttributeInt(
                ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL
            )) {
                ExifInterface.ORIENTATION_ROTATE_90 -> 90f
                ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                else -> 0f
            }
        } catch (_: Exception) {
            0f
        }

        return if (rotation != 0f) {
            val matrix = Matrix().apply { postRotate(rotation) }
            Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
        } else {
            bmp
        }
    }

    override fun onDestroy() {
        executor.shutdown()
        interpreter?.close()
        interpreter = null
        super.onDestroy()
    }
}
