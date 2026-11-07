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

    private val lock = Any()
    private var pendingPhoto: Bitmap? = null
    private var pendingDepth: Bitmap? = null
    private var curPhoto: Bitmap? = null
    private var curDepth: Bitmap? = null
    private var needUpload = false

    private var prog = 0
    private val tex = IntArray(2)
    private var locScale = 0
    private var locTilt = 0
    private var locPhoto = 0
    private var locDepth = 0
    private var vw = 1
    private var vh = 1
    private var imgAspect = 1f

    private val quad: FloatBuffer = ByteBuffer.allocateDirect(8 * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
            put(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)); position(0)
        }

    fun setImages(photo: Bitmap, depth: Bitmap) {
        synchronized(lock) { pendingPhoto = photo; pendingDepth = depth }
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        prog = link(VERT, FRAG)
        locScale = GLES20.glGetUniformLocation(prog, "uScale")
        locTilt = GLES20.glGetUniformLocation(prog, "uTilt")
        locPhoto = GLES20.glGetUniformLocation(prog, "uPhoto")
        locDepth = GLES20.glGetUniformLocation(prog, "uDepth")
        GLES20.glGenTextures(2, tex, 0)
        needUpload = curPhoto != null // контекст пересоздан — перезалить текстуры
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        vw = width; vh = height
        GLES20.glViewport(0, 0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        synchronized(lock) {
            if (pendingPhoto != null && pendingDepth != null) {
                curPhoto = pendingPhoto; curDepth = pendingDepth
                pendingPhoto = null; pendingDepth = null
                needUpload = true
            }
        }
        val photo = curPhoto ?: return
        val depth = curDepth ?: return
        if (needUpload) {
            upload(tex[0], photo)
            upload(tex[1], depth)
            imgAspect = photo.width.toFloat() / photo.height
            needUpload = false
        }

        // "cover": заполняем экран, лишнее обрезаем + запас под сдвиг
        val viewAspect = vw.toFloat() / vh
        var sx = 1f
        var sy = 1f
        if (imgAspect > viewAspect) sx = viewAspect / imgAspect else sy = imgAspect / viewAspect
        val margin = 1f - strength * 2.2f
        sx *= margin; sy *= margin

        GLES20.glUseProgram(prog)
        GLES20.glUniform2f(locScale, sx, sy)
        GLES20.glUniform2f(locTilt, tiltX * strength, tiltY * strength)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex[0])
        GLES20.glUniform1i(locPhoto, 0)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, tex[1])
        GLES20.glUniform1i(locDepth, 1)

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
        private const val FRAG = """
            precision mediump float;
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
    }
}
