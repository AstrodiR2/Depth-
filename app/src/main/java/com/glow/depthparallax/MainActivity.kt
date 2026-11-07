package com.glow.depthparallax

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.ImageDecoder
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.net.Uri
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import java.util.concurrent.atomic.AtomicInteger
import kotlin.concurrent.thread
import kotlin.math.max

class MainActivity : ComponentActivity(), SensorEventListener {

    private lateinit var gl: GLSurfaceView
    private lateinit var status: TextView
    private lateinit var pickBtn: Button
    private lateinit var layerLabel: TextView
    private lateinit var seek: SeekBar
    private val renderer = ParallaxRenderer()

    private lateinit var sm: SensorManager
    private var sign = 1f

    private var curX = 0f
    private var curZ = 0f
    private var baseX = 0f
    private var baseZ = 0f
    private var firstEvent = true

    private var photo: Bitmap? = null
    private var depth: Bitmap? = null
    private var depthInfo = ""
    private var layerCount = 8
    private val gen = AtomicInteger()

    private val picker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) process(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        sm = getSystemService(SENSOR_SERVICE) as SensorManager

        gl = GLSurfaceView(this).apply {
            setEGLContextClientVersion(2)
            setRenderer(renderer)
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
            setOnClickListener { firstEvent = true } // тап = перецентровать
        }

        status = TextView(this).apply {
            setTextColor(Color.WHITE)
            setShadowLayer(6f, 0f, 0f, Color.BLACK)
            textSize = 14f
            text = "Выбери фото"
            setPadding(32, 64, 32, 0)
        }

        layerLabel = TextView(this).apply {
            setTextColor(Color.WHITE)
            setShadowLayer(6f, 0f, 0f, Color.BLACK)
            textSize = 15f
            gravity = Gravity.CENTER
            text = "Слоёв: $layerCount"
        }
        seek = SeekBar(this).apply {
            max = 39
            progress = layerCount - 1
            setPadding(64, 8, 64, 8)
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar, p: Int, fromUser: Boolean) {
                    layerLabel.text = "Слоёв: ${p + 1}"
                }
                override fun onStartTrackingTouch(sb: SeekBar) {}
                override fun onStopTrackingTouch(sb: SeekBar) {
                    layerCount = sb.progress + 1
                    rebuild()
                }
            })
        }

        pickBtn = Button(this).apply {
            text = "Выбрать фото"
            setOnClickListener { picker.launch("image/*") }
        }
        val flipBtn = Button(this).apply {
            text = "Инверт"
            setOnClickListener { sign = -sign }
        }
        val buttons = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            addView(pickBtn)
            addView(flipBtn)
        }
        val panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(layerLabel)
            addView(seek, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
            addView(buttons)
        }

        val root = FrameLayout(this)
        root.addView(gl, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
        root.addView(status, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.TOP))
        root.addView(panel, FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM
        ).apply { bottomMargin = 96 })
        setContentView(root)
    }

    private fun process(uri: Uri) {
        pickBtn.isEnabled = false
        status.text = "Считаю глубину…"
        thread {
            try {
                val p = decode(uri)
                val res = DepthEstimator(this).estimate(p)
                val refined = DepthRefiner.refine(p, res.depth)
                photo = p
                depth = refined
                depthInfo = "${res.backend}, ${res.ms} мс"
                runOnUiThread {
                    pickBtn.isEnabled = true
                    rebuild()
                }
            } catch (e: Throwable) {
                runOnUiThread {
                    status.text = "Ошибка: ${e.javaClass.simpleName}: ${e.message}"
                    pickBtn.isEnabled = true
                }
            }
        }
    }

    /** Пересобирает слои под текущее число (фото и глубина уже посчитаны). */
    private fun rebuild() {
        val p = photo ?: return
        val d = depth ?: return
        val n = layerCount
        val g = gen.incrementAndGet()
        status.text = "Строю слои ($n)…"
        thread {
            try {
                val t0 = System.currentTimeMillis()
                val scene = if (n == 1) {
                    ParallaxRenderer.Scene(p, d, emptyList(), FloatArray(0))
                } else {
                    val r = LayerBuilder.build(p, d, n)
                    ParallaxRenderer.Scene(p, d, r.bitmaps, r.shifts)
                }
                if (g != gen.get()) return@thread
                gl.queueEvent { renderer.setScene(scene) }
                val ms = System.currentTimeMillis() - t0
                val layersText = if (n == 1) "1 (плавная глубина)" else "${scene.layers.size} из $n"
                runOnUiThread {
                    status.text = "Глубина: $depthInfo\nСлоёв: $layersText, $ms мс\nНаклоняй телефон. Тап — центр."
                }
            } catch (e: Throwable) {
                runOnUiThread { status.text = "Ошибка слоёв: ${e.javaClass.simpleName}: ${e.message}" }
            }
        }
    }

    private fun decode(uri: Uri): Bitmap {
        val src = ImageDecoder.createSource(contentResolver, uri)
        return ImageDecoder.decodeBitmap(src) { decoder, info, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            val m = max(info.size.width, info.size.height)
            if (m > 2048) {
                val s = 2048f / m
                decoder.setTargetSize((info.size.width * s).toInt(), (info.size.height * s).toInt())
            }
        }
    }

    override fun onResume() {
        super.onResume()
        gl.onResume()
        sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)?.let {
            sm.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        firstEvent = true
    }

    override fun onPause() {
        super.onPause()
        gl.onPause()
        sm.unregisterListener(this)
    }

    override fun onSensorChanged(e: SensorEvent) {
        val x = e.values[0] / 9.81f
        val z = e.values[2] / 9.81f
        if (firstEvent) {
            curX = x; curZ = z; baseX = x; baseZ = z
            firstEvent = false
        }
        curX += (x - curX) * 0.15f
        curZ += (z - curZ) * 0.15f
        // база медленно подтягивается — эффект сам центрируется
        baseX += (curX - baseX) * 0.003f
        baseZ += (curZ - baseZ) * 0.003f
        renderer.tiltX = (sign * (curX - baseX) / 0.3f).coerceIn(-1f, 1f)
        renderer.tiltY = (sign * (curZ - baseZ) / 0.3f).coerceIn(-1f, 1f)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
