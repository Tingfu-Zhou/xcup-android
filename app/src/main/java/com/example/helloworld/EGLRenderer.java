package com.example.helloworld;

import android.graphics.SurfaceTexture;
import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLSurface;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * 离线模式抽帧的 GPU 部分：MediaCodec 把解码出的帧渲染到 {@link #getSurfaceTexture()}（OES 外部纹理），
 * 这里按 SurfaceTexture 的变换矩阵（含解码器的裁边、旋转、上下翻转）把“要送进模型的区域”画到一张 size×size 的
 * 离屏 pbuffer 上，再读回 CPU（之后由 {@link FramePreprocessor} 做面积缩小）。
 *
 * <p>2026-10-14 之前：不用变换矩阵（靠手写的翻转纹理坐标），整帧一步画到 160×160，没有裁剪也没有抗混叠，
 * 并且渲染后 sleep(20) 赌帧已经到了。现在等 onFrameAvailable 并核对时间戳，确认拿到的就是刚渲染的那一帧。</p>
 *
 * <p>EGL 上下文绑定在创建它的线程上：构造、awaitFrame、drawAndRead、release 必须都在同一个线程调用
 * （VideoFrameExtractor 的抽帧线程）。onFrameAvailable 回调走单独的小线程，所以抽帧线程可以阻塞等待。</p>
 */
final class EGLRenderer {

    private static final String TAG = "EGLRenderer";

    private static final String VERTEX_SHADER =
            "uniform mat4 uTexMatrix;\n" +
            "attribute vec4 aPosition;\n" +
            "attribute vec4 aTexCoord;\n" +
            "varying vec2 vTexCoord;\n" +
            "void main() {\n" +
            "    gl_Position = aPosition;\n" +
            "    vTexCoord = (uTexMatrix * aTexCoord).xy;\n" +
            "}\n";

    // 纹理坐标用 highp：mediump（fp16）在 1920 宽的纹理上误差接近 1 个像素
    private static final String FRAGMENT_SHADER =
            "#extension GL_OES_EGL_image_external : require\n" +
            "#ifdef GL_FRAGMENT_PRECISION_HIGH\n" +
            "precision highp float;\n" +
            "#else\n" +
            "precision mediump float;\n" +
            "#endif\n" +
            "varying vec2 vTexCoord;\n" +
            "uniform samplerExternalOES uTexture;\n" +
            "void main() {\n" +
            "    gl_FragColor = texture2D(uTexture, vTexCoord);\n" +
            "}\n";

    // 全屏四边形（TRIANGLE_STRIP：左下、右下、左上、右上）
    private static final float[] VERTICES = {-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f};

    private final int size;
    private final EGLDisplay eglDisplay;
    private final EGLContext eglContext;
    private final EGLSurface eglSurface;
    private final int oesTextureId;
    private final int program;
    private final int aPosition, aTexCoord, uTexMatrix, uTexture;
    private final FloatBuffer vertexBuffer;
    private final FloatBuffer texCoordBuffer;
    private final float[] texMatrix = new float[16];
    private final ByteBuffer readBuffer;
    private final int[] argb;

    private final SurfaceTexture surfaceTexture;
    private final HandlerThread callbackThread;
    private final Object frameLock = new Object();
    private int framesAvailable = 0;           // onFrameAvailable 计数，awaitFrame 消费
    private boolean warnedTimestamp = false;

    /** @param size 离屏 pbuffer 的边长（输出 size×size） */
    EGLRenderer(int size) {
        this.size = size;

        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) throw new RuntimeException("eglGetDisplay 失败");
        int[] ver = new int[2];
        if (!EGL14.eglInitialize(eglDisplay, ver, 0, ver, 1)) throw eglError("eglInitialize");

        int[] attribList = {
                EGL14.EGL_RED_SIZE, 8,
                EGL14.EGL_GREEN_SIZE, 8,
                EGL14.EGL_BLUE_SIZE, 8,
                EGL14.EGL_ALPHA_SIZE, 8,
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,     // 默认只要求 WINDOW_BIT，有的机型建不了 pbuffer
                EGL14.EGL_NONE
        };
        EGLConfig[] configs = new EGLConfig[1];
        int[] numConfigs = new int[1];
        if (!EGL14.eglChooseConfig(eglDisplay, attribList, 0, configs, 0, 1, numConfigs, 0)
                || numConfigs[0] <= 0) {
            throw eglError("eglChooseConfig");
        }
        int[] contextAttribs = {EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE};
        eglContext = EGL14.eglCreateContext(eglDisplay, configs[0], EGL14.EGL_NO_CONTEXT, contextAttribs, 0);
        if (eglContext == null || eglContext == EGL14.EGL_NO_CONTEXT) throw eglError("eglCreateContext");
        int[] surfaceAttribs = {EGL14.EGL_WIDTH, size, EGL14.EGL_HEIGHT, size, EGL14.EGL_NONE};
        eglSurface = EGL14.eglCreatePbufferSurface(eglDisplay, configs[0], surfaceAttribs, 0);
        if (eglSurface == null || eglSurface == EGL14.EGL_NO_SURFACE) throw eglError("eglCreatePbufferSurface");
        if (!EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)) throw eglError("eglMakeCurrent");

        int[] textures = new int[1];
        GLES20.glGenTextures(1, textures, 0);
        oesTextureId = textures[0];
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTextureId);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        program = createProgram(VERTEX_SHADER, FRAGMENT_SHADER);
        aPosition = GLES20.glGetAttribLocation(program, "aPosition");
        aTexCoord = GLES20.glGetAttribLocation(program, "aTexCoord");
        uTexMatrix = GLES20.glGetUniformLocation(program, "uTexMatrix");
        uTexture = GLES20.glGetUniformLocation(program, "uTexture");

        vertexBuffer = ByteBuffer.allocateDirect(VERTICES.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        vertexBuffer.put(VERTICES).position(0);
        texCoordBuffer = ByteBuffer.allocateDirect(8 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        setRegion(0f, 0f, 1f, 1f);
        readBuffer = ByteBuffer.allocateDirect(size * size * 4).order(ByteOrder.nativeOrder());
        argb = new int[size * size];

        // onFrameAvailable 默认投递到创建 SurfaceTexture 的线程的 Looper（这里就是抽帧线程），而抽帧线程要阻塞等这一帧，
        // 会死锁。所以回调交给单独的小线程，只负责计数和唤醒。
        callbackThread = new HandlerThread("FrameAvailableCb");
        callbackThread.start();
        surfaceTexture = new SurfaceTexture(oesTextureId);
        surfaceTexture.setOnFrameAvailableListener(st -> {
            synchronized (frameLock) {
                framesAvailable++;
                frameLock.notifyAll();
            }
        }, new Handler(callbackThread.getLooper()));
    }

    SurfaceTexture getSurfaceTexture() {
        return surfaceTexture;
    }

    /**
     * 要画的区域，逻辑图像坐标（0..1，已经过变换矩阵摆正；原点在左下角）。画到整个 pbuffer 上。
     */
    void setRegion(float s0, float t0, float s1, float t1) {
        float[] tc = {s0, t0, s1, t0, s0, t1, s1, t1};
        texCoordBuffer.clear();
        texCoordBuffer.put(tc).position(0);
    }

    /**
     * 等解码器刚渲染的那一帧到达并锁定到纹理（updateTexImage）。
     *
     * @param expectedTimestampNs 这一帧的时间戳：MediaCodec 渲染时用 pts（微秒）× 1000
     * @return false 表示超时没等到
     */
    boolean awaitFrame(long expectedTimestampNs, long timeoutMs) {
        long deadline = System.nanoTime() + timeoutMs * 1_000_000L;
        boolean latchedOther = false;
        while (true) {
            synchronized (frameLock) {
                while (framesAvailable == 0) {
                    long leftNs = deadline - System.nanoTime();
                    if (leftNs <= 0) {
                        // 拿到过一帧但时间戳对不上，之后又没有新帧：多半是这台机器改写了时间戳，就用这一帧
                        if (latchedOther && !warnedTimestamp) {
                            warnedTimestamp = true;
                            Log.w(TAG, "SurfaceTexture 时间戳与 pts 对不上（期望 " + expectedTimestampNs
                                    + "，实际 " + surfaceTexture.getTimestamp() + "），按最新一帧处理");
                        }
                        return latchedOther;
                    }
                    try {
                        frameLock.wait(Math.max(1L, leftNs / 1_000_000L));
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return false;
                    }
                }
                framesAvailable--;
            }
            surfaceTexture.updateTexImage();              // 锁定最新一帧（更早的会被丢掉）
            if (surfaceTexture.getTimestamp() == expectedTimestampNs) {
                // 每次只渲染一帧：锁定了它，剩下的计数都是更早、已被丢掉的帧
                synchronized (frameLock) {
                    framesAvailable = 0;
                }
                return true;
            }
            latchedOther = true;                          // 之前超时没等到的旧帧，或时间戳被改写：再等等
            deadline = Math.min(deadline, System.nanoTime() + 30_000_000L);
        }
    }

    /**
     * 把当前锁定的帧画到 pbuffer 上并读回。
     *
     * @return size×size 的 ARGB 像素（自上而下）。数组会被下一次调用复用，用完前不要再调
     */
    int[] drawAndRead() {
        surfaceTexture.getTransformMatrix(texMatrix);
        GLES20.glViewport(0, 0, size, size);
        GLES20.glClearColor(0f, 0f, 0f, 1f);
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
        GLES20.glUseProgram(program);
        GLES20.glEnableVertexAttribArray(aPosition);
        GLES20.glVertexAttribPointer(aPosition, 2, GLES20.GL_FLOAT, false, 0, vertexBuffer);
        GLES20.glEnableVertexAttribArray(aTexCoord);
        GLES20.glVertexAttribPointer(aTexCoord, 2, GLES20.GL_FLOAT, false, 0, texCoordBuffer);
        GLES20.glUniformMatrix4fv(uTexMatrix, 1, false, texMatrix, 0);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, oesTextureId);
        GLES20.glUniform1i(uTexture, 0);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        readBuffer.rewind();
        GLES20.glReadPixels(0, 0, size, size, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, readBuffer);
        readBuffer.rewind();
        FramePreprocessor.rgbaBottomUpToArgb(readBuffer, size, size, argb);
        return argb;
    }

    void release() {
        surfaceTexture.setOnFrameAvailableListener(null);
        surfaceTexture.release();
        callbackThread.quitSafely();
        GLES20.glDeleteProgram(program);
        GLES20.glDeleteTextures(1, new int[]{oesTextureId}, 0);
        EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT);
        EGL14.eglDestroySurface(eglDisplay, eglSurface);
        EGL14.eglDestroyContext(eglDisplay, eglContext);
        EGL14.eglReleaseThread();
        EGL14.eglTerminate(eglDisplay);
    }

    private static RuntimeException eglError(String what) {
        return new RuntimeException(what + " 失败: 0x" + Integer.toHexString(EGL14.eglGetError()));
    }

    private static int createProgram(String vertexSource, String fragmentSource) {
        int vs = loadShader(GLES20.GL_VERTEX_SHADER, vertexSource);
        int fs = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentSource);
        int program = GLES20.glCreateProgram();
        GLES20.glAttachShader(program, vs);
        GLES20.glAttachShader(program, fs);
        GLES20.glLinkProgram(program);
        int[] ok = new int[1];
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, ok, 0);
        if (ok[0] != GLES20.GL_TRUE) {
            String log = GLES20.glGetProgramInfoLog(program);
            GLES20.glDeleteProgram(program);
            throw new RuntimeException("着色器链接失败: " + log);
        }
        return program;
    }

    private static int loadShader(int type, String source) {
        int shader = GLES20.glCreateShader(type);
        GLES20.glShaderSource(shader, source);
        GLES20.glCompileShader(shader);
        int[] ok = new int[1];
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, ok, 0);
        if (ok[0] == 0) {
            String log = GLES20.glGetShaderInfoLog(shader);
            GLES20.glDeleteShader(shader);
            throw new RuntimeException("着色器编译失败: " + log);
        }
        return shader;
    }
}
