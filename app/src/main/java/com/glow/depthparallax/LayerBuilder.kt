package com.glow.depthparallax

import android.graphics.Bitmap
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Нарезает фото на N слоёв по глубине (от дальнего к ближнему).
 *
 * Слой j = "сцена без всего, что ближе слоя j", где скрытое достроено (push-pull).
 * Альфа слоя j = 1 там, где глубина >= начала слоя (сам слой + всё ближе, это перекроют ближние слои).
 * Каждый слой двигается со своей скоростью, и на месте ушедшего объекта виден достроенный фон.
 */
object LayerBuilder {

    class Result(val bitmaps: List<Bitmap>, val shifts: FloatArray)

    fun build(photo: Bitmap, depthBmp: Bitmap, n: Int): Result {
        val pw = photo.width
        val ph = photo.height
        val aspect = pw.toDouble() / ph

        // разрешение слоя: общий бюджет ~24 Мпикс на все слои
        val px = min(24_000_000.0 / n, pw.toDouble() * ph)
        val lh = min(ph, Math.round(sqrt(px / aspect)).toInt().coerceAtLeast(64))
        val lw = min(pw, Math.round(aspect * lh).toInt().coerceAtLeast(64))
        val total = lw * lh

        val small = Bitmap.createScaledBitmap(photo, lw, lh, true)
        val spx = IntArray(total).also { small.getPixels(it, 0, lw, 0, 0, lw, lh) }
        if (small !== photo) small.recycle()
        val rgb = FloatArray(total * 3)
        for (i in 0 until total) {
            val p = spx[i]
            rgb[i * 3] = ((p shr 16) and 255) / 255f
            rgb[i * 3 + 1] = ((p shr 8) and 255) / 255f
            rgb[i * 3 + 2] = (p and 255) / 255f
        }

        // глубина в разрешение слоя (билинейно)
        val dw = depthBmp.width
        val dh = depthBmp.height
        val dpx = IntArray(dw * dh).also { depthBmp.getPixels(it, 0, dw, 0, 0, dw, dh) }
        val dsrc = FloatArray(dw * dh) { (dpx[it] and 255) / 255f }
        val d = FloatArray(total)
        for (y in 0 until lh) {
            val fy = (y + 0.5f) / lh * dh - 0.5f
            val y0 = floor(fy).toInt()
            val ty = fy - y0
            val ya = y0.coerceIn(0, dh - 1)
            val yb = (y0 + 1).coerceIn(0, dh - 1)
            for (x in 0 until lw) {
                val fx = (x + 0.5f) / lw * dw - 0.5f
                val x0 = floor(fx).toInt()
                val tx = fx - x0
                val xa = x0.coerceIn(0, dw - 1)
                val xb = (x0 + 1).coerceIn(0, dw - 1)
                d[y * lw + x] =
                    dsrc[ya * dw + xa] * (1 - tx) * (1 - ty) + dsrc[ya * dw + xb] * tx * (1 - ty) +
                    dsrc[yb * dw + xa] * (1 - tx) * ty + dsrc[yb * dw + xb] * tx * ty
            }
        }

        // нормируем глубину по 2%..98%, чтобы слои распределились по реальному диапазону сцены
        val hist = IntArray(256)
        for (v in d) hist[(v * 255f).toInt().coerceIn(0, 255)]++
        var acc = 0
        var lo = 0
        var hi = 255
        for (i in 0 until 256) { acc += hist[i]; if (acc >= total * 0.02f) { lo = i; break } }
        acc = 0
        for (i in 0 until 256) { acc += hist[i]; if (acc >= total * 0.98f) { hi = i; break } }
        val loV = lo / 255f
        val hiV = max(hi / 255f, loV + 0.05f)
        for (i in 0 until total) d[i] = ((d[i] - loV) / (hiV - loV)).coerceIn(0f, 1f)

        // пустые срезы глубины пропускаем (лишние проходы отрисовки)
        val cnt = IntArray(n)
        for (v in d) cnt[min(n - 1, (v * n).toInt())]++
        val minCount = max(50, (total * 0.0005f).toInt())
        val keep = (0 until n).filter { it == 0 || cnt[it] >= minCount }

        val soft = min(0.03f, 0.35f / n)
        val r = max(2, lw / 300)
        val workers = (200_000_000.0 / (total * 35.0)).toInt().coerceIn(1, 4)

        val pool = Executors.newFixedThreadPool(workers)
        try {
            val tasks = keep.map { j ->
                Callable {
                    val hole = FloatArray(total)
                    if (j < n - 1) {
                        val b1 = (j + 1f) / n
                        for (i in 0 until total) hole[i] = smooth(b1 - soft, b1 + soft, d[i])
                    }
                    dilateInPlace(hole, lw, lh, r)
                    for (i in 0 until total) hole[i] = 1f - min(1f, hole[i] * 3f)
                    val filled = pushPull(rgb, hole, lw, lh, 3)

                    val bj = j.toFloat() / n
                    val out = IntArray(total)
                    for (i in 0 until total) {
                        val a = if (j == 0) 1f else smooth(bj - soft, bj + soft, d[i])
                        out[i] = (gv(a) shl 24) or (gv(filled[i * 3]) shl 16) or
                            (gv(filled[i * 3 + 1]) shl 8) or gv(filled[i * 3 + 2])
                    }
                    Bitmap.createBitmap(out, lw, lh, Bitmap.Config.ARGB_8888)
                }
            }
            val bmps = pool.invokeAll(tasks).map { it.get() }
            val shifts = FloatArray(keep.size) { (keep[it] + 0.5f) / n }
            return Result(bmps, shifts)
        } finally {
            pool.shutdown()
        }
    }

    private fun gv(v: Float) = (v * 255f + 0.5f).toInt().coerceIn(0, 255)

    private fun smooth(e0: Float, e1: Float, x: Float): Float {
        val t = ((x - e0) / (e1 - e0)).coerceIn(0f, 1f)
        return t * t * (3f - 2f * t)
    }

    private fun dilateInPlace(a: FloatArray, w: Int, h: Int, r: Int) {
        val tmp = FloatArray(a.size)
        for (y in 0 until h) for (x in 0 until w) {
            var mx = 0f
            for (k in -r..r) mx = max(mx, a[y * w + (x + k).coerceIn(0, w - 1)])
            tmp[y * w + x] = mx
        }
        for (y in 0 until h) for (x in 0 until w) {
            var mx = 0f
            for (k in -r..r) mx = max(mx, tmp[(y + k).coerceIn(0, h - 1) * w + x])
            a[y * w + x] = mx
        }
    }

    private class Lvl(val w: Int, val h: Int, val c: FloatArray, val wt: FloatArray)

    /** Push-pull: заполняет пиксели с весом 0 значениями ближайших валидных. Входные массивы не меняет. */
    private fun pushPull(col: FloatArray, wt: FloatArray, w: Int, h: Int, ch: Int): FloatArray {
        val levels = ArrayList<Lvl>()
        levels.add(Lvl(w, h, col.copyOf(), wt.copyOf()))
        val sum = FloatArray(ch)
        while (levels.last().w > 1 || levels.last().h > 1) {
            val p = levels.last()
            val nw = (p.w + 1) / 2
            val nh = (p.h + 1) / 2
            val c = FloatArray(nw * nh * ch)
            val wn = FloatArray(nw * nh)
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
