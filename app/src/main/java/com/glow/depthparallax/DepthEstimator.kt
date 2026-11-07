package com.glow.depthparallax

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.EnumSet

/** ZipDepth (6.1M параметров, ONNX FP16). Вход RGB 0..1, выход — обратная глубина (ближе = больше). */
class DepthEstimator(private val ctx: Context) {

    data class Result(val depth: Bitmap, val backend: String, val ms: Long)

    private val size = 512 // кратно 32. Можно 384 (быстрее) или 640 (детальнее)

    private val modelBytes: ByteArray by lazy {
        ctx.assets.open("zipdepth.onnx").use { it.readBytes() }
    }

    fun estimate(photo: Bitmap): Result {
        val t0 = System.currentTimeMillis()
        return try {
            val bmp = run(photo, true)
            Result(bmp, "NNAPI (NPU/GPU)", System.currentTimeMillis() - t0)
        } catch (e: Throwable) {
            Log.w("Depth", "NNAPI failed, fallback to CPU", e)
            val t1 = System.currentTimeMillis()
            val bmp = run(photo, false)
            Result(bmp, "CPU (NNAPI не сработал)", System.currentTimeMillis() - t1)
        }
    }

    private fun run(photo: Bitmap, useNnapi: Boolean): Bitmap {
        val env = OrtEnvironment.getEnvironment()
        val so = OrtSession.SessionOptions()
        so.addFreeDimensionOverrideByName("batch_size", 1)
        so.addFreeDimensionOverrideByName("height", size.toLong())
        so.addFreeDimensionOverrideByName("width", size.toLong())
        if (useNnapi) so.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16)) else so.setIntraOpNumThreads(4)

        val session = env.createSession(modelBytes, so)
        try {
            val scaled = Bitmap.createScaledBitmap(photo, size, size, true)
            val px = IntArray(size * size)
            scaled.getPixels(px, 0, size, 0, 0, size, size)
            if (scaled !== photo) scaled.recycle()

            val n = size * size
            val buf = ByteBuffer.allocateDirect(3 * n * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
            for (i in 0 until n) buf.put(i, ((px[i] shr 16) and 255) / 255f)
            for (i in 0 until n) buf.put(n + i, ((px[i] shr 8) and 255) / 255f)
            for (i in 0 until n) buf.put(2 * n + i, (px[i] and 255) / 255f)
            buf.rewind()

            val inputName = session.inputNames.iterator().next()
            OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, size.toLong(), size.toLong())).use { input ->
                session.run(mapOf(inputName to input)).use { res ->
                    val out = res.get(0) as OnnxTensor
                    val shape = out.info.shape
                    val dims = shape.filter { it > 1 }
                    val oh = if (dims.size >= 2) dims[dims.size - 2].toInt() else size
                    val ow = if (dims.size >= 2) dims[dims.size - 1].toInt() else size
                    val fb = out.floatBuffer
                    val arr = FloatArray(fb.remaining())
                    fb.get(arr)

                    // отсекаем выбросы (1%..99%), чтобы контраст не "съедался"
                    val sorted = arr.copyOf().also { it.sort() }
                    val lo = sorted[(sorted.size * 0.01f).toInt()]
                    val hi = sorted[(sorted.size * 0.99f).toInt().coerceAtMost(sorted.size - 1)]
                    val range = if (hi - lo < 1e-6f) 1f else hi - lo

                    val pixels = IntArray(ow * oh)
                    for (i in pixels.indices) {
                        val g = (((arr[i] - lo) / range) * 255f).toInt().coerceIn(0, 255)
                        pixels[i] = (0xFF shl 24) or (g shl 16) or (g shl 8) or g
                    }
                    return Bitmap.createBitmap(pixels, ow, oh, Bitmap.Config.ARGB_8888)
                }
            }
        } finally {
            session.close()
            so.close()
        }
    }
}
