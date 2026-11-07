package com.glow.depthparallax

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class DepthEstimator(private val ctx: Context) {

    data class Result(val depth: Bitmap, val backend: String, val ms: Long)

    private val model: ByteBuffer by lazy {
        ctx.assets.openFd("midas_small.tflite").use { fd ->
            FileInputStream(fd.fileDescriptor).channel.map(
                FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
            )
        }
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
        val opts = Interpreter.Options()
        var nn: NnApiDelegate? = null
        if (useNnapi) {
            nn = NnApiDelegate()
            opts.addDelegate(nn)
        } else {
            opts.setNumThreads(4)
        }
        val interp = Interpreter(model, opts)
        try {
            val inShape = interp.getInputTensor(0).shape() // [1, H, W, 3]
            val h = inShape[1]
            val w = inShape[2]

            val scaled = Bitmap.createScaledBitmap(photo, w, h, true)
            val px = IntArray(w * h)
            scaled.getPixels(px, 0, w, 0, 0, w, h)
            if (scaled !== photo) scaled.recycle()

            val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
            val std = floatArrayOf(0.229f, 0.224f, 0.225f)
            val input = ByteBuffer.allocateDirect(w * h * 3 * 4).order(ByteOrder.nativeOrder())
            for (p in px) {
                input.putFloat((((p shr 16) and 255) / 255f - mean[0]) / std[0])
                input.putFloat((((p shr 8) and 255) / 255f - mean[1]) / std[1])
                input.putFloat(((p and 255) / 255f - mean[2]) / std[2])
            }
            input.rewind()

            val outTensor = interp.getOutputTensor(0)
            val outN = outTensor.numElements()
            val dims = outTensor.shape().filter { it > 1 }
            val oh = if (dims.size >= 2) dims[dims.size - 2] else h
            val ow = if (dims.size >= 2) dims[dims.size - 1] else w

            val out = ByteBuffer.allocateDirect(outN * 4).order(ByteOrder.nativeOrder())
            interp.run(input, out)
            out.rewind()
            val arr = FloatArray(outN)
            out.asFloatBuffer().get(arr)

            var mn = Float.MAX_VALUE
            var mx = -Float.MAX_VALUE
            for (v in arr) { if (v < mn) mn = v; if (v > mx) mx = v }
            val range = if (mx - mn < 1e-6f) 1f else mx - mn

            val pixels = IntArray(ow * oh)
            for (i in pixels.indices) {
                val g = (((arr[i] - mn) / range) * 255f).toInt().coerceIn(0, 255)
                pixels[i] = (0xFF shl 24) or (g shl 16) or (g shl 8) or g
            }
            return Bitmap.createBitmap(pixels, ow, oh, Bitmap.Config.ARGB_8888)
        } finally {
            interp.close()
            nn?.close()
        }
    }
}
