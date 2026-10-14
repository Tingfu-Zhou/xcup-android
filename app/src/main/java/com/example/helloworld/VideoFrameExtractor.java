package com.example.helloworld;

import android.content.Context;
import android.graphics.Bitmap;
import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;
import android.util.Log;
import android.view.Surface;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Map;

/**
 * 离线模式的抽帧器：给一个播放位置，返回<b>这一刻的那一帧</b>，已经按 {@link FramePreprocessor} 处理成模型输入（160×160）。
 *
 * <p><b>2026-10-14 之前的问题</b>：每次都 {@code seekTo(T, SEEK_TO_CLOSEST_SYNC)}，然后直接返回解码器吐出的第一帧，
 * 拿到的其实是离 T 最近的关键帧（可能在 T 之后，最多差半个 GOP）。关键帧间隔 4~12 秒的片子，12 帧窗口里只有 1~2 张
 * 不同的图，视频结果每个 GOP 才变一次（bench 的 v08~v13 上每次冻住 4.5~8.5 秒）。</p>
 *
 * <p><b>现在的做法</b>：
 * <ul>
 *   <li>只在必要时重新定位（第一次、{@link #requestSeek()} 之后、播放位置往回跳、或往前跳太远）：
 *       {@code seekTo(T, SEEK_TO_PREVIOUS_SYNC)} + flush，从 T 之前的关键帧往后解；</li>
 *   <li>平时顺着往后解（每次只多解约 250ms 的帧），pts 不到 T 的帧直接丢掉、不渲染；取 pts 不晚于 T + 半帧的最后一帧
 *       （即离 T 最近的那一帧）渲染。第一个越过的输出 buffer 先扣住，留给下一次；</li>
 *   <li>每次最多花 {@link #BUDGET_MS}。seek 进长 GOP 后来不及解到 T 时返回 null，下次接着解（追赶），
 *       不拿离 T 太远的帧凑数。VideoView 的 seek 本身就落在前一个关键帧上，所以通常不用追。</li>
 * </ul></p>
 *
 * <p><b>线程</b>：EGL 上下文绑定在线程上，所以 {@link #getFrameAt}、{@link #requestSeek}、{@link #release} 必须在同一个线程调用
 * （VideoProcessActivity 的抽帧线程）。构造函数只打开 MediaExtractor、读格式，可以在任意线程调用；EGL 和解码器在第一次
 * getFrameAt 时创建。</p>
 */
public class VideoFrameExtractor {

    private static final String TAG = "VideoFrameExtractor";

    /** GPU 先把要送进模型的区域画成 STAGE_SIZE×STAGE_SIZE（3×224，1080p 时只缩小 1.6 倍，双线性基本不混叠），再在 CPU 上面积缩小。 */
    public static final int STAGE_SIZE = 672;

    /** 每次 getFrameAt 最多花这么久解码（追赶时分几次解完）。 */
    private static final long BUDGET_MS = 150;
    /** 追赶中（还没解到目标）时，候选帧离目标超过这么久就不要。 */
    private static final long STALE_US = 200_000;
    /** 目标比已显示的帧早这么多以上，才当作往回跳（播放位置偶尔会往回抖几十毫秒）。 */
    private static final long BACK_TOL_US = 100_000;
    /** 已经追上之后目标又往前跳了这么多，重新定位比顺着解快。 */
    private static final long FORWARD_SEEK_US = 3_000_000;
    private static final long DEQUEUE_TIMEOUT_US = 2_000;
    private static final long FRAME_WAIT_MS = 100;
    private static final int MAX_FAILURES = 5;

    /** 一次取帧的结果：模型输入和这一帧的 pts。 */
    public static final class Frame {
        public final Bitmap bitmap;
        public final long ptsUs;

        Frame(Bitmap bitmap, long ptsUs) {
            this.bitmap = bitmap;
            this.ptsUs = ptsUs;
        }
    }

    private final MediaExtractor extractor;
    private final MediaFormat format;
    private final String mime;
    private final long frameDurUs;
    private final float[] region;             // {s0, t0, s1, t1}：逻辑图像坐标，原点左下

    private EGLRenderer renderer;
    private Surface surface;
    private MediaCodec decoder;
    private final MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();

    private boolean needSeek = true;
    private boolean inputDone, outputDone;
    private boolean inputSinceStart;          // start() 之后喂过输入
    private boolean outputSinceStart;         // start() 之后拿到过输出（或格式变化）：此后 flush 才不会丢掉 csd
    private int heldIndex = -1;               // 扣住的输出 buffer：第一个越过上次目标的帧
    private long heldPtsUs = -1;
    private long decodedPtsUs = -1;           // 最后一个出队的输出帧的 pts（渲染、丢弃、扣住都算）
    private long seekTargetUs = 0;            // 最近一次定位的目标
    private long shownPtsUs = -1;             // 最近一次渲染、返回的帧
    private Bitmap shownBitmap;
    private int failures = 0;
    private boolean broken = false;
    private boolean released = false;

    public VideoFrameExtractor(Context context, Uri videoUri) throws IOException {
        this(context, videoUri, null);
    }

    public VideoFrameExtractor(Context context, Uri videoUri, Map<String, String> headers) throws IOException {
        extractor = new MediaExtractor();
        extractor.setDataSource(context, videoUri, headers);

        MediaFormat videoFormat = null;
        for (int i = 0; i < extractor.getTrackCount(); i++) {
            MediaFormat f = extractor.getTrackFormat(i);
            String m = f.getString(MediaFormat.KEY_MIME);
            if (m != null && m.startsWith("video/")) {
                extractor.selectTrack(i);
                videoFormat = f;
                break;
            }
        }
        if (videoFormat == null) {
            extractor.release();
            throw new IOException("未找到视频轨道");
        }
        format = videoFormat;
        mime = format.getString(MediaFormat.KEY_MIME);
        frameDurUs = frameDurationUs(format);

        // 显示尺寸（带旋转的片子宽高互换）：解码器按 format 里的 rotation 把画面转正，变换矩阵里已经包含
        int w = format.getInteger(MediaFormat.KEY_WIDTH), h = format.getInteger(MediaFormat.KEY_HEIGHT);
        int rot = format.containsKey(MediaFormat.KEY_ROTATION) ? format.getInteger(MediaFormat.KEY_ROTATION) : 0;
        if (rot % 180 != 0) {
            int t = w;
            w = h;
            h = t;
        }
        int[] r = FramePreprocessor.cropRect(w, h);          // {x0, y0, cw, ch}，原点左上
        region = new float[]{
                r[0] / (float) w, (h - r[1] - r[3]) / (float) h,
                (r[0] + r[2]) / (float) w, (h - r[1]) / (float) h};
        Log.i(TAG, String.format("视频 %s %dx%d rot=%d 帧长 %dus，送进模型的区域 %s", mime, w, h, rot, frameDurUs,
                FramePreprocessor.CENTER_CROP ? "居中 " + r[2] + "x" + r[3] : "整帧"));
    }

    private static long frameDurationUs(MediaFormat f) {
        float fps = 0f;
        if (f.containsKey(MediaFormat.KEY_FRAME_RATE)) {
            try {
                fps = f.getInteger(MediaFormat.KEY_FRAME_RATE);
            } catch (ClassCastException e) {
                fps = f.getFloat(MediaFormat.KEY_FRAME_RATE);
            }
        }
        if (fps < 1f || fps > 240f) fps = 30f;
        return Math.round(1_000_000.0 / fps);
    }

    /** 下一次 getFrameAt 按它的目标重新定位（用户 seek 之后调用）。只能在抽帧线程调用。 */
    public void requestSeek() {
        needSeek = true;
    }

    /**
     * 取播放位置 targetUs 处的那一帧。只能在抽帧线程调用。
     *
     * @return 这一帧（模型输入 + pts）；还在追赶、等帧超时或出错时返回 null
     */
    public Frame getFrameAt(long targetUs) {
        if (released || broken) return null;
        try {
            ensureStarted();
            Frame f = step(targetUs);
            failures = 0;
            return f;
        } catch (RuntimeException | IOException e) {          // CodecException 是 IllegalStateException；EGL 失败是 RuntimeException
            Log.e(TAG, "取帧失败，重建解码器后从目标位置重新开始", e);
            if (++failures >= MAX_FAILURES) {
                broken = true;
                Log.e(TAG, "连续失败 " + failures + " 次，不再取帧");
            } else {
                recreateDecoder();
            }
            return null;
        }
    }

    private Frame step(long targetUs) throws IOException {
        long limit = targetUs + frameDurUs / 2;               // 取 pts <= limit 的最后一帧 = 离目标最近的一帧
        boolean back = shownPtsUs >= 0 && targetUs < shownPtsUs - BACK_TOL_US;
        boolean jumped = decodedPtsUs >= seekTargetUs && targetUs > decodedPtsUs + FORWARD_SEEK_US;
        if (needSeek || back || jumped) seekInternal(targetUs);
        if (decoder == null) return null;                     // 重建解码器失败

        int candIndex = -1;
        long candPts = -1;
        if (heldIndex >= 0 && heldPtsUs <= limit) {          // 上次扣住的帧，这次轮到它了
            candIndex = heldIndex;
            candPts = heldPtsUs;
            heldIndex = -1;
        }
        long deadline = System.nanoTime() + BUDGET_MS * 1_000_000L;
        while (heldIndex < 0 && !outputDone) {
            feedInput();
            int idx = decoder.dequeueOutputBuffer(info, DEQUEUE_TIMEOUT_US);
            if (idx >= 0) {
                outputSinceStart = true;
                boolean eos = (info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
                // 有的解码器在 EOS buffer 里带一个无效的时间戳：时间倒退的 EOS buffer 不当成帧
                if (info.size > 0 && !(eos && info.presentationTimeUs < decodedPtsUs)) {
                    long pts = info.presentationTimeUs;
                    decodedPtsUs = pts;
                    if (pts <= limit) {
                        if (candIndex >= 0) decoder.releaseOutputBuffer(candIndex, false);  // 丢掉更早的候选，不渲染
                        candIndex = idx;
                        candPts = pts;
                    } else {
                        heldIndex = idx;                      // 第一帧越过目标：扣住，留给下一次
                        heldPtsUs = pts;
                    }
                } else {
                    decoder.releaseOutputBuffer(idx, false);  // 只带 EOS 标记的空 buffer
                }
                if (eos) outputDone = true;
            } else if (idx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                outputSinceStart = true;
            }
            if (System.nanoTime() > deadline) break;
        }

        if (candIndex >= 0) {
            boolean caughtUp = heldIndex >= 0 || outputDone;
            if (!caughtUp && targetUs - candPts > STALE_US) {
                decoder.releaseOutputBuffer(candIndex, false);
                Log.d(TAG, String.format("追赶中：已解到 %dms，目标 %dms", candPts / 1000, targetUs / 1000));
                return null;
            }
            return render(candIndex, candPts);
        }
        if (shownBitmap != null && (heldIndex >= 0 || outputDone)) {
            return new Frame(shownBitmap, shownPtsUs);       // 目标还没走到下一帧（或片尾）：仍是刚才那一帧
        }
        if (heldIndex >= 0 && shownBitmap == null) {          // 刚定位完，第一帧就已经在目标之后（目标早于第一个关键帧）
            int idx = heldIndex;
            heldIndex = -1;
            return render(idx, heldPtsUs);
        }
        return null;
    }

    private Frame render(int index, long ptsUs) {
        decoder.releaseOutputBuffer(index, true);             // 时间戳 = pts × 1000，awaitFrame 按它核对
        if (!renderer.awaitFrame(ptsUs * 1000L, FRAME_WAIT_MS)) {
            Log.w(TAG, "等 " + (ptsUs / 1000) + "ms 那一帧超时，跳过");
            return null;
        }
        int[] staged = renderer.drawAndRead();
        int[] px = FramePreprocessor.toModelInput(staged, STAGE_SIZE, STAGE_SIZE, null);  // 区域已在 GPU 上取好
        Bitmap bmp = Bitmap.createBitmap(px, FramePreprocessor.MODEL_SIZE, FramePreprocessor.MODEL_SIZE,
                Bitmap.Config.ARGB_8888);
        shownBitmap = bmp;
        shownPtsUs = ptsUs;
        return new Frame(bmp, ptsUs);
    }

    private void feedInput() {
        while (!inputDone) {
            int inIdx = decoder.dequeueInputBuffer(0);
            if (inIdx < 0) return;
            ByteBuffer buf = decoder.getInputBuffer(inIdx);
            int size = buf == null ? -1 : extractor.readSampleData(buf, 0);
            if (size < 0) {
                decoder.queueInputBuffer(inIdx, 0, 0, 0L, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                inputDone = true;
            } else {
                decoder.queueInputBuffer(inIdx, 0, size, extractor.getSampleTime(), 0);
                extractor.advance();
            }
            inputSinceStart = true;
        }
    }

    private void seekInternal(long targetUs) throws IOException {
        heldIndex = -1;                                       // flush / 重建会收回所有输出 buffer
        extractor.seekTo(targetUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC);
        if (inputSinceStart) {
            if (outputSinceStart) {
                decoder.flush();
            } else {
                // start() 之后还没出过输出就 flush，会把 csd（SPS/PPS 等）一起冲掉，之后解不出来；这种情况重建解码器
                recreateDecoder();
            }
        }
        inputDone = false;
        outputDone = false;
        decodedPtsUs = -1;
        shownPtsUs = -1;
        shownBitmap = null;
        seekTargetUs = targetUs;
        needSeek = false;
        Log.d(TAG, String.format("定位到 %dms：从 %dms 的关键帧开始解", targetUs / 1000,
                Math.max(0, extractor.getSampleTime()) / 1000));
    }

    private void ensureStarted() throws IOException {
        if (decoder != null) return;
        if (renderer == null) {
            renderer = new EGLRenderer(STAGE_SIZE);
            renderer.setRegion(region[0], region[1], region[2], region[3]);
            surface = new Surface(renderer.getSurfaceTexture());
        }
        decoder = MediaCodec.createDecoderByType(mime);
        decoder.configure(format, surface, null, 0);
        decoder.start();
        inputSinceStart = false;
        outputSinceStart = false;
    }

    /** 出错后（或 start 之后还没出过输出就要定位时）重建解码器；下一次 getFrameAt 从目标位置重新定位。 */
    private void recreateDecoder() {
        if (decoder != null) {
            try {
                decoder.stop();
            } catch (Exception ignored) {
            }
            try {
                decoder.release();
            } catch (Exception ignored) {
            }
            decoder = null;
        }
        heldIndex = -1;
        needSeek = true;
        try {
            ensureStarted();
        } catch (Exception e) {
            Log.e(TAG, "重建解码器失败", e);
            broken = true;
        }
    }

    /** 释放解码器、EGL 等。只能在抽帧线程调用。 */
    public void release() {
        released = true;
        if (decoder != null) {
            try {
                decoder.stop();
            } catch (Exception ignored) {
            }
            try {
                decoder.release();
            } catch (Exception ignored) {
            }
            decoder = null;
        }
        extractor.release();
        if (surface != null) surface.release();
        if (renderer != null) renderer.release();
    }
}
