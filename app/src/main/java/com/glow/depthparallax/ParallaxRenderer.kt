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

    @Volatile var tiltX = 0f   // -1..1
    @Volatile var tiltY = 0f   // -1..1
    @Volatile var strength = 0.04f

    // порядок: photo, depth, bgPhoto, bgDepth, mask
    private val lock = Any()
    private var pending: Array<Bitmap>? = null
    private var cur: Array<Bitmap>? = null
    private var needUpload = false

    private var prog = 0
    private val tex = IntArray(5)
    private var locScale = 0
    private var locTilt = 0
    private val locSamplers = IntArray(5)
    private var vw = 1
    private var vh = 1
    private var imgAspect = 1f

    private val quad: FloatBuffer = ByteBuffer.allocateDirect(8 * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
            put(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)); position(0)
        }

    fun setImages(photo: Bitmap, depth: Bitmap, bgPhoto: Bitmap, bgDepth: Bitmap, mask: Bitmap) {
        synchronized(lock) { pending = arrayOf(photo, depth, bgPhoto, bgDepth, mask) }
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        prog = link(VERT, FRAG)
        locScale = GLES20.glGetUniformLocation(prog, "uScale")
        locTilt = GLES20.glGetUniformLocation(prog, "uTilt")
        val names = arrayOf("uPhoto", "uDepth", "uBg", "uBgDepth", "uMask")
        for (i in 0 until 5) locSamplers[i] = GLES20.glGetUniformLocation(prog, names[i])
        GLES20.glGenTextures(5, tex, 0)
        needUpload = cur != null // контекст пересоздан — перезалить текстуры
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        vw = width; vh = height
        GLES20.glViewport(0, 0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        synchronized(lock) {
            pending?.let { cur = it; pending = null; needUpload = true }
        }
        val imgs = cur ?: return
        if (needUpload) {
            for (i in 0 until 5) upload(tex[i], imgs[i])
            imgAspect = imgs[0].width.toFloat() / imgs[0].height
            needUpload = false
        }

        val viewAspect = vw.toFloat() / vh
        var sx = 1f
        var sy = 1f
        if (imgAspect > viewAspect) sx = viewAspect / imgAspect else sy = imgAspect / viewAspect
        val margin = 1f - strength * 2.2f
        sx *= margin; sy *= margin

        GLES20.glUseProgram(prog)
        GLES20.glUniform2f(locScale, sx, sy)
        GLES20.glUniform2f(locTilt, tiltX * strength, tiltY * strength)
        for (i in 0 until 5) {
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0 + i)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex[i])
            GLES20.glUniform1i(locSamplers[i], i)
        }

        val aPos = GLES20.glGetAttribLocation(prog, "aPos")
        GLES20.glEnableVertexAttribArray(aPos)
        GLES20.glVertexAttribPointer(aPos, 2, GLES20.GL_FLOAT, false, 0, quad)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(aPos)
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
        // Два слоя: объект (по маске) и достроенный фон. Каждый сдвигается по своей глубине.
        private const val FRAG = """
            precision highp float;
            varying vec2 vUv;
            uniform sampler2D uPhoto;
            uniform sampler2D uDepth;
            uniform sampler2D uBg;
            uniform sampler2D uBgDepth;
            uniform sampler2D uMask;
            uniform vec2 uTilt;
            void main() {
                // объект: стартуем с предположения "близко", чтобы он корректно наезжал на фон
                vec2 uf = clamp(vUv + uTilt * 0.5, 0.0, 1.0);
                for (int i = 0; i < 3; i++) {
                    float d = texture2D(uDepth, uf).r;
                    uf = clamp(vUv + uTilt * (d - 0.5), 0.0, 1.0);
                }
                float a = texture2D(uMask, uf).r;
                vec3 fg = texture2D(uPhoto, uf).rgb;

                vec2 ub = vUv;
                for (int i = 0; i < 2; i++) {
                    float db = texture2D(uBgDepth, ub).r;
                    ub = clamp(vUv + uTilt * (db - 0.5), 0.0, 1.0);
                }
                vec3 bg = texture2D(uBg, ub).rgb;

                gl_FragColor = vec4(mix(bg, fg, a), 1.0);
            }
        """
    }
}
