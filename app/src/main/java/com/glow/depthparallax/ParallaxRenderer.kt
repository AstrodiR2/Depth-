package com.glow.depthparallax

import android.graphics.Bitmap
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.GLUtils
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class ParallaxRenderer : GLSurfaceView.Renderer {

    /**
     * layers пусто -> режим "1 слой": плавная глубина (photo + depth).
     * иначе слои от дальнего к ближнему, shifts[i] = глубина слоя 0..1.
     */
    class Scene(val photo: Bitmap, val depth: Bitmap, val layers: List<Bitmap>, val shifts: FloatArray)

    @Volatile var tiltX = 0f   // -1..1
    @Volatile var tiltY = 0f   // -1..1
    @Volatile var strength = 0.04f

    private val lock = Any()
    private var pending: Scene? = null
    private var cur: Scene? = null
    private var needUpload = false

    private var texIds = IntArray(0)

    private var progFlat = 0
    private var locFScale = 0
    private var locFTilt = 0
    private var locFPhoto = 0
    private var locFDepth = 0

    private var progLayer = 0
    private var locLScale = 0
    private var locLShift = 0
    private var locLSampler = 0

    private var vw = 1
    private var vh = 1
    private var imgAspect = 1f

    private val quad: FloatBuffer = ByteBuffer.allocateDirect(8 * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
            put(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)); position(0)
        }

    fun setScene(s: Scene) {
        synchronized(lock) { pending = s }
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        progFlat = link(VERT, FRAG_FLAT)
        locFScale = GLES20.glGetUniformLocation(progFlat, "uScale")
        locFTilt = GLES20.glGetUniformLocation(progFlat, "uTilt")
        locFPhoto = GLES20.glGetUniformLocation(progFlat, "uPhoto")
        locFDepth = GLES20.glGetUniformLocation(progFlat, "uDepth")

        progLayer = link(VERT, FRAG_LAYER)
        locLScale = GLES20.glGetUniformLocation(progLayer, "uScale")
        locLShift = GLES20.glGetUniformLocation(progLayer, "uShift")
        locLSampler = GLES20.glGetUniformLocation(progLayer, "uLayer")

        texIds = IntArray(0) // контекст новый — старые id недействительны
        needUpload = cur != null
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        vw = width; vh = height
        GLES20.glViewport(0, 0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

        var old: Scene? = null
        synchronized(lock) {
            pending?.let { old = cur; cur = it; pending = null; needUpload = true }
        }
        val scene = cur ?: return
        if (needUpload) {
            if (texIds.isNotEmpty()) GLES20.glDeleteTextures(texIds.size, texIds, 0)
            val bmps = if (scene.layers.isEmpty()) listOf(scene.photo, scene.depth) else scene.layers
            texIds = IntArray(bmps.size)
            GLES20.glGenTextures(bmps.size, texIds, 0)
            for (i in bmps.indices) upload(texIds[i], bmps[i])
            imgAspect = scene.photo.width.toFloat() / scene.photo.height
            needUpload = false
            // слои прошлой сцены больше не нужны
            old?.layers?.forEach { if (!it.isRecycled) it.recycle() }
        }

        val viewAspect = vw.toFloat() / vh
        var sx = 1f
        var sy = 1f
        if (imgAspect > viewAspect) sx = viewAspect / imgAspect else sy = imgAspect / viewAspect
        val margin = 1f - strength * 2.2f
        sx *= margin; sy *= margin

        if (scene.layers.isEmpty()) drawFlat(sx, sy) else drawLayers(scene, sx, sy)
    }

    private fun drawFlat(sx: Float, sy: Float) {
        GLES20.glUseProgram(progFlat)
        GLES20.glUniform2f(locFScale, sx, sy)
        GLES20.glUniform2f(locFTilt, tiltX * strength, tiltY * strength)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texIds[0])
        GLES20.glUniform1i(locFPhoto, 0)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texIds[1])
        GLES20.glUniform1i(locFDepth, 1)
        val aPos = GLES20.glGetAttribLocation(progFlat, "aPos")
        GLES20.glEnableVertexAttribArray(aPos)
        GLES20.glVertexAttribPointer(aPos, 2, GLES20.GL_FLOAT, false, 0, quad)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(aPos)
    }

    private fun drawLayers(scene: Scene, sx: Float, sy: Float) {
        GLES20.glUseProgram(progLayer)
        GLES20.glUniform2f(locLScale, sx, sy)
        GLES20.glUniform1i(locLSampler, 0)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA) // текстуры с премультиплицированной альфой
        val aPos = GLES20.glGetAttribLocation(progLayer, "aPos")
        GLES20.glEnableVertexAttribArray(aPos)
        GLES20.glVertexAttribPointer(aPos, 2, GLES20.GL_FLOAT, false, 0, quad)
        for (i in scene.layers.indices) {
            val k = scene.shifts[i] - 0.5f
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texIds[i])
            GLES20.glUniform2f(locLShift, tiltX * strength * k, tiltY * strength * k)
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        }
        GLES20.glDisableVertexAttribArray(aPos)
        GLES20.glDisable(GLES20.GL_BLEND)
    }

    private fun upload(id: Int, bmp: Bitmap) {
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, id)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bmp, 0)
    }

    private fun link(v: String, f: String): Int {
        fun compile(type: Int, src: String): Int {
            val s = GLES20.glCreateShader(type)
            GLES20.glShaderSource(s, src)
            GLES20.glCompileShader(s)
            val ok = IntArray(1)
            GLES20.glGetShaderiv(s, GLES20.GL_COMPILE_STATUS, ok, 0)
            if (ok[0] == 0) error("Shader: " + GLES20.glGetShaderInfoLog(s))
            return s
        }
        val p = GLES20.glCreateProgram()
        GLES20.glAttachShader(p, compile(GLES20.GL_VERTEX_SHADER, v))
        GLES20.glAttachShader(p, compile(GLES20.GL_FRAGMENT_SHADER, f))
        GLES20.glLinkProgram(p)
        return p
    }

    companion object {
        private const val VERT = """
            attribute vec2 aPos;
            uniform vec2 uScale;
            varying vec2 vUv;
            void main() {
                vec2 t = aPos * 0.5 + 0.5;
                t.y = 1.0 - t.y;
                vUv = (t - 0.5) * uScale + 0.5;
                gl_Position = vec4(aPos, 0.0, 1.0);
            }
        """
        // 1 слой: плавная глубина, каждый пиксель сдвигается по своей глубине
        private const val FRAG_FLAT = """
            precision highp float;
            varying vec2 vUv;
            uniform sampler2D uPhoto;
            uniform sampler2D uDepth;
            uniform vec2 uTilt;
            void main() {
                vec2 uv = vUv;
                for (int i = 0; i < 3; i++) {
                    float d = texture2D(uDepth, clamp(uv, 0.0, 1.0)).r;
                    uv = vUv + uTilt * (d - 0.5);
                }
                gl_FragColor = texture2D(uPhoto, clamp(uv, 0.0, 1.0));
            }
        """
        // N слоёв: каждый слой целиком сдвигается на свою величину
        private const val FRAG_LAYER = """
            precision highp float;
            varying vec2 vUv;
            uniform sampler2D uLayer;
            uniform vec2 uShift;
            void main() {
                gl_FragColor = texture2D(uLayer, clamp(vUv + uShift, 0.0, 1.0));
            }
        """
    }
}
