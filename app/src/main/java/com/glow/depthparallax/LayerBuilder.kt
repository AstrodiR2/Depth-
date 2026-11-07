package com.glow.depthparallax

import android.graphics.Bitmap
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Делит фото на два слоя: объект (по маске из глубины) и фон.
 * Фон под объектом достраивается методом push-pull (заливка из соседних пикселей),
 * поэтому при наклоне на месте объекта открывается фон, а не растянутая картинка.
 */
object LayerBuilder {

    class Layers(val depth: Bitmap, val bgPhoto: Bitmap, val bgDepth: Bitmap, val mask: Bitmap)

    fun build(photo: Bitmap, depthBmp: Bitmap): Layers {
        val w = depthBmp.width
        val h = depthBmp.height
        val n = w * h
        val dpx = IntArray(n).also { depthBmp.getPixels(it, 0, w, 0, 0, w, h) }
        val d = FloatArray(n) { (dpx[it] and 255) / 255f }

        // 1. маска объекта: порог по Оцу + заливка дыр внутри объекта (блики хрома и т.п.)
        val t = otsu(d).coerceIn(0.3f, 0.7f)
        val band = 0.08f
        val m = FloatArray(n) { smooth(t - band, t + band, d[it]) }
        fillHoles(m, w, h)

        // 2. глубина объекта: внутри маски не даём падать ниже порога, чтобы объект не "рвался"
        val lift = min(1f, t + 0.1f)
        val depthAdj = FloatArray(n) { val di = d[it]; di + (max(di, lift) - di) * m[it] }

        // 3. фон: убираем объект (с запасом по краю) и достраиваем
        val dil = dilate(m, w, h, 3)
        val wgt = FloatArray(n) { 1f - min(1f, dil[it] * 3f) }

        val small = Bitmap.createScaledBitmap(photo, w, h, true)
        val spx = IntArray(n).also { small.getPixels(it, 0, w, 0, 0, w, h) }
        if (small !== photo) small.recycle()
        val data = FloatArray(n * 4)
        for (i in 0 until n) {
            val p = spx[i]
            data[i * 4] = ((p shr 16) and 255) / 255f
            data[i * 4 + 1] = ((p shr 8) and 255) / 255f
            data[i * 4 + 2] = (p and 255) / 255f
            data[i * 4 + 3] = d[i]
        }
        val filled = pushPull(data, wgt, w, h, 4)

        // 4. полноразмерный фон: оригинал везде, кроме места под объектом
        val pw = photo.width
        val ph = photo.height
        val ppx = IntArray(pw * ph).also { photo.getPixels(it, 0, pw, 0, 0, pw, ph) }
        for (y in 0 until ph) {
            val fy = (y + 0.5f) / ph * h - 0.5f
            val y0 = floor(fy).toInt()
            val ty = fy - y0
            val ya = y0.coerceIn(0, h - 1)
            val yb = (y0 + 1).coerceIn(0, h - 1)
            for (x in 0 until pw) {
                val fx = (x + 0.5f) / pw * w - 0.5f
                val x0 = floor(fx).toInt()
                val tx = fx - x0
                val xa = x0.coerceIn(0, w - 1)
                val xb = (x0 + 1).coerceIn(0, w - 1)
                val i00 = ya * w + xa
                val i10 = ya * w + xb
                val i01 = yb * w + xa
                val i11 = yb * w + xb
                val w00 = (1 - tx) * (1 - ty)
                val w10 = tx * (1 - ty)
                val w01 = (1 - tx) * ty
                val w11 = tx * ty
                val a = min(1f, (dil[i00] * w00 + dil[i10] * w10 + dil[i01] * w01 + dil[i11] * w11) * 3f)
                if (a <= 0.001f) continue
                val p = ppx[y * pw + x]
                var r = ((p shr 16) and 255) / 255f
                var g = ((p shr 8) and 255) / 255f
                var b = (p and 255) / 255f
                val fr = filled[i00 * 4] * w00 + filled[i10 * 4] * w10 + filled[i01 * 4] * w01 + filled[i11 * 4] * w11
                val fg = filled[i00 * 4 + 1] * w00 + filled[i10 * 4 + 1] * w10 + filled[i01 * 4 + 1] * w01 + filled[i11 * 4 + 1] * w11
                val fb = filled[i00 * 4 + 2] * w00 + filled[i10 * 4 + 2] * w10 + filled[i01 * 4 + 2] * w01 + filled[i11 * 4 + 2] * w11
                r += (fr - r) * a
                g += (fg - g) * a
                b += (fb - b) * a
                ppx[y * pw + x] = (0xFF shl 24) or
                    ((r * 255f).toInt().coerceIn(0, 255) shl 16) or
                    ((g * 255f).toInt().coerceIn(0, 255) shl 8) or
                    (b * 255f).toInt().coerceIn(0, 255)
            }
        }
        val bgPhoto = Bitmap.createBitmap(ppx, pw, ph, Bitmap.Config.ARGB_8888)
        val bgDepth = gray(IntArray(n) { gv(filled[it * 4 + 3]) }, w, h)
        val depth = gray(IntArray(n) { gv(depthAdj[it]) }, w, h)
        val mask = gray(IntArray(n) { gv(m[it]) }, w, h)
        return Layers(depth, bgPhoto, bgDepth, mask)
    }

    private fun gv(v: Float) = (v * 255f).toInt().coerceIn(0, 255)

    private fun gray(g: IntArray, w: Int, h: Int): Bitmap {
        val px = IntArray(g.size) { (0xFF shl 24) or (g[it] shl 16) or (g[it] shl 8) or g[it] }
        return Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)
    }

    private fun smooth(e0: Float, e1: Float, x: Float): Float {
        val t = ((x - e0) / (e1 - e0)).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }

    private fun otsu(d: FloatArray): Float {
        val hist = IntArray(256)
        for (v in d) hist[(v * 255f).toInt().coerceIn(0, 255)]++
        val total = d.size.toFloat()
        var sumAll = 0f
        for (i in 0 until 256) sumAll += i * hist[i]
        var sumB = 0f
        var wB = 0f
        var best = 0f
        var bestT = 127
        for (i in 0 until 256) {
            wB += hist[i]
            if (wB == 0f) continue
            val wF = total - wB
            if (wF == 0f) break
            sumB += i * hist[i]
            val mB = sumB / wB
            val mF = (sumAll - sumB) / wF
            val between = wB * wF * (mB - mF) * (mB - mF)
            if (between > best) { best = between; bestT = i }
        }
        return bestT / 255f
    }

    /** Всё, что в маске не связано с краем кадра "не-объектом", считаем объектом (заливка дыр). */
    private fun fillHoles(m: FloatArray, w: Int, h: Int) {
        val n = w * h
        val bin = BooleanArray(n) { m[it] > 0.5f }
        val reach = BooleanArray(n)
        val q = IntArray(n)
        var head = 0
        var tail = 0
        fun seed(i: Int) { if (!bin[i] && !reach[i]) { reach[i] = true; q[tail++] = i } }
        for (x in 0 until w) { seed(x); seed((h - 1) * w + x) }
        for (y in 0 until h) { seed(y * w); seed(y * w + w - 1) }
        while (head < tail) {
            val i = q[head++]
            val x = i % w
            val y = i / w
            if (x > 0) seed(i - 1)
            if (x < w - 1) seed(i + 1)
            if (y > 0) seed(i - w)
            if (y < h - 1) seed(i + w)
        }
        for (i in 0 until n) if (!bin[i] && !reach[i]) m[i] = 1f
    }

    private fun dilate(src: FloatArray, w: Int, h: Int, r: Int): FloatArray {
        val tmp = FloatArray(src.size)
        for (y in 0 until h) for (x in 0 until w) {
            var mx = 0f
            for (k in -r..r) mx = max(mx, src[y * w + (x + k).coerceIn(0, w - 1)])
            tmp[y * w + x] = mx
        }
        val out = FloatArray(src.size)
        for (y in 0 until h) for (x in 0 until w) {
            var mx = 0f
            for (k in -r..r) mx = max(mx, tmp[(y + k).coerceIn(0, h - 1) * w + x])
            out[y * w + x] = mx
        }
        return out
    }

    private class Lvl(val w: Int, val h: Int, val c: FloatArray, val wt: FloatArray)

    /** Push-pull: заполняет пиксели с весом 0 значениями из ближайших валидных. */
    private fun pushPull(col: FloatArray, wt: FloatArray, w: Int, h: Int, ch: Int): FloatArray {
        val levels = ArrayList<Lvl>()
        levels.add(Lvl(w, h, col.copyOf(), wt.copyOf()))
        while (levels.last().w > 1 || levels.last().h > 1) {
            val p = levels.last()
            val nw = (p.w + 1) / 2
            val nh = (p.h + 1) / 2
            val c = FloatArray(nw * nh * ch)
            val wn = FloatArray(nw * nh)
            val sum = FloatArray(ch)
            for (y in 0 until nh) for (x in 0 until nw) {
                var sw = 0f
                var cnt = 0
                java.util.Arrays.fill(sum, 0f)
                for (dy in 0..1) for (dx in 0..1) {
                    val px = 2 * x + dx
                    val py = 2 * y + dy
                    if (px >= p.w || py >= p.h) continue
                    cnt++
                    val idx = py * p.w + px
                    val wv = p.wt[idx]
                    sw += wv
                    for (k in 0 until ch) sum[k] += wv * p.c[idx * ch + k]
                }
                val o = y * nw + x
                if (sw > 1e-6f) {
                    for (k in 0 until ch) c[o * ch + k] = sum[k] / sw
                    wn[o] = min(1f, sw / cnt)
                }
            }
            levels.add(Lvl(nw, nh, c, wn))
        }
        val up = FloatArray(ch)
        for (l in levels.size - 2 downTo 0) {
            val f = levels[l]
            val cs = levels[l + 1]
            for (y in 0 until f.h) for (x in 0 until f.w) {
                val i = y * f.w + x
                val wv = f.wt[i]
                if (wv >= 0.999f) continue
                val fx = (x + 0.5f) / 2f - 0.5f
                val fy = (y + 0.5f) / 2f - 0.5f
                val x0 = floor(fx).toInt()
                val y0 = floor(fy).toInt()
                val tx = fx - x0
                val ty = fy - y0
                val xa = x0.coerceIn(0, cs.w - 1)
                val xb = (x0 + 1).coerceIn(0, cs.w - 1)
                val ya = y0.coerceIn(0, cs.h - 1)
                val yb = (y0 + 1).coerceIn(0, cs.h - 1)
                for (k in 0 until ch) {
                    up[k] = cs.c[(ya * cs.w + xa) * ch + k] * (1 - tx) * (1 - ty) +
                        cs.c[(ya * cs.w + xb) * ch + k] * tx * (1 - ty) +
                        cs.c[(yb * cs.w + xa) * ch + k] * (1 - tx) * ty +
                        cs.c[(yb * cs.w + xb) * ch + k] * tx * ty
                    f.c[i * ch + k] = wv * f.c[i * ch + k] + (1 - wv) * up[k]
                }
            }
        }
        return levels[0].c
    }
}
