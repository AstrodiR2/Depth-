package com.glow.depthparallax

import android.graphics.Bitmap
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Joint bilateral upsampling: растягивает грубую карту глубины (256x256)
 * до разрешения фото и "прилепляет" границы глубины к границам на самом фото.
 * Из-за этого пропадает размазывание картинки вокруг объекта.
 */
object DepthRefiner {

    fun refine(photo: Bitmap, depth: Bitmap, maxSide: Int = 768): Bitmap {
        val s = min(1f, maxSide.toFloat() / max(photo.width, photo.height))
        val tw = max(1, (photo.width * s).toInt())
        val th = max(1, (photo.height * s).toInt())
        val dw = depth.width
        val dh = depth.height

        val small = Bitmap.createScaledBitmap(photo, tw, th, true)
        val guideBmp = Bitmap.createScaledBitmap(photo, dw, dh, true)
        val sp = IntArray(tw * th).also { small.getPixels(it, 0, tw, 0, 0, tw, th) }
        val gp = IntArray(dw * dh).also { guideBmp.getPixels(it, 0, dw, 0, 0, dw, dh) }
        val dp = IntArray(dw * dh).also { depth.getPixels(it, 0, dw, 0, 0, dw, dh) }
        if (small !== photo) small.recycle()
        if (guideBmp !== photo) guideBmp.recycle()

        val d = FloatArray(dw * dh) { (dp[it] and 255) / 255f }
        val gr = FloatArray(dw * dh) { ((gp[it] shr 16) and 255) / 255f }
        val gg = FloatArray(dw * dh) { ((gp[it] shr 8) and 255) / 255f }
        val gb = FloatArray(dw * dh) { (gp[it] and 255) / 255f }

        val sigmaS2 = 2f * 1.0f * 1.0f      // пространственный (в текселях глубины)
        val sigmaR2 = 2f * 0.10f * 0.10f    // по цвету (0..1)

        val out = IntArray(tw * th)
        for (y in 0 until th) {
            val fy = (y + 0.5f) / th * dh - 0.5f
            val iy = floor(fy).toInt()
            for (x in 0 until tw) {
                val fx = (x + 0.5f) / tw * dw - 0.5f
                val ix = floor(fx).toInt()
                val p = sp[y * tw + x]
                val pr = ((p shr 16) and 255) / 255f
                val pg = ((p shr 8) and 255) / 255f
                val pb = (p and 255) / 255f

                var sum = 0f
                var wsum = 0f
                for (j in iy - 1..iy + 2) {
                    val yy = j.coerceIn(0, dh - 1)
                    for (i in ix - 1..ix + 2) {
                        val xx = i.coerceIn(0, dw - 1)
                        val k = yy * dw + xx
                        val dx = i - fx
                        val dy = j - fy
                        val dr = pr - gr[k]
                        val dg = pg - gg[k]
                        val db = pb - gb[k]
                        val w = exp(-(dx * dx + dy * dy) / sigmaS2 - (dr * dr + dg * dg + db * db) / sigmaR2)
                        sum += w * d[k]
                        wsum += w
                    }
                }
                val v = if (wsum > 1e-6f) sum / wsum
                else d[iy.coerceIn(0, dh - 1) * dw + ix.coerceIn(0, dw - 1)]
                val g = (v * 255f).toInt().coerceIn(0, 255)
                out[y * tw + x] = (0xFF shl 24) or (g shl 16) or (g shl 8) or g
            }
        }
        return Bitmap.createBitmap(out, tw, th, Bitmap.Config.ARGB_8888)
    }
}
