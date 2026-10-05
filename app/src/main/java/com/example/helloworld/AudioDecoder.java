package com.example.helloworld;

import android.content.Context;
import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;
import android.util.Log;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;
import java.util.Map;
import java.util.function.Supplier;


/**
 * AudioDecoder 类：
 * AudioDecoder只负责实时解码并把 PCM 数据送入 PcmCircularBuffer
 */
public class AudioDecoder {

    private static final String TAG = "AudioDecoder";
    private Supplier<Integer> videoPositionProvider;

    // 7.24新增：seek后宽限期相关常量
    private static final long SEEK_GRACE_PERIOD = 2000; // seek后的宽限期2秒
    private static final long MAX_ADVANCE_NORMAL = 3000; // 正常情况下提前3秒
    private static final long MAX_ADVANCE_AFTER_SEEK = 3000; // seek后允许提前3秒
    private volatile long lastSeekTimeInDecoder = 0; // 记录解码器中的seek时间


    private Context context;
    private Uri videoUri;
    private Map<String, String> httpHeaders;   // 网页视频模式下使用的 HTTP headers
    private AudioInferenceHelper audioHelper;
    private volatile boolean isDecoding = false;

    // 音频分析完成监听器（用于主界面记录音频耗时）
    private Runnable onCompleteListener = null;

    private PcmCircularBuffer pcmBuffer;

    /**
     * [音频修复] true：每个解码 buffer 交给 {@link PcmResampler}（按声道下混 + 抗混叠重采样到 16kHz），
     * 连续写入、不丢样本，模型窗口就是真正的 [T-2s, T]。
     * false：原来的处理——交错 PCM 当单声道、每 4 秒块只用前一半，双声道片源送进模型的是 0.5 倍速音频
     * （bench/tools/audio_rescore.py check 在 6 部片上证实）。保留它是因为现有 fusion_config.json 的证据表
     * 是在这种音频上拟合的：用哪种，由配置的 audio_pipeline 决定（VideoProcessActivity 读）。
     */
    private final boolean fixedPipeline;
    /** 两个相邻解码 buffer 的 pts 与按样本数推算的时间差超过这个，就当作不连续，重采样器重新起头。 */
    private static final long DISCONTINUITY_US = 100_000;

    private volatile boolean isDecodingFinished = false; // 新增：标志位，表示解码是否完成

    private int seekToPositionMs = -1;


    /**
     * 供外部通知 decoder 跳转位置
     */
    // 可接受误差范围（毫秒）
    private static final int SEEK_POSITION_TOLERANCE_MS = 200;

    private long lastSeekRealTime = 0;

    public void seekTo(int ms) {
        long now = System.currentTimeMillis();
        if ((now - lastSeekRealTime) < 1000) {
            Log.w(TAG, "⚠️ seekTo 频率过快，距上次不足 1000ms，跳过");
            return;
        }
        lastSeekRealTime = now;

        // 7.24新增：记录seek时间，供解码线程使用
        lastSeekTimeInDecoder = now;

        Log.d(TAG, "🔄 AudioDecoder.seekTo() called, new time: " + ms + "ms");

        stop();
        this.seekToPositionMs = ms;  // 直接seek到当前位置，让限速机制自然控制提前量。
        startDecoding();
        Log.d(TAG, "🚀 seekTo() -> startDecoding(), seekToPositionMs = " + seekToPositionMs);

    }


    public AudioDecoder(Context context, Uri videoUri, PcmCircularBuffer buffer, Supplier<Integer> videoPositionProvider) {
        this(context, videoUri, null, buffer, videoPositionProvider);
    }

    public AudioDecoder(Context context, Uri videoUri, Map<String, String> httpHeaders,
                        PcmCircularBuffer buffer, Supplier<Integer> videoPositionProvider) {
        this(context, videoUri, httpHeaders, buffer, videoPositionProvider, false);
    }

    public AudioDecoder(Context context, Uri videoUri, Map<String, String> httpHeaders,
                        PcmCircularBuffer buffer, Supplier<Integer> videoPositionProvider,
                        boolean fixedPipeline) {
        this.context = context;
        this.videoUri = videoUri;
        this.httpHeaders = httpHeaders;
        this.pcmBuffer = buffer;
        this.videoPositionProvider = videoPositionProvider;
        this.fixedPipeline = fixedPipeline;
    }


    /**
     * 设置音频分析完成后的回调
     */
    public void setOnCompleteListener(Runnable listener) {
        this.onCompleteListener = listener;
    }

    /**
     * 开始解码音频（在子线程中执行）
     */
    public void startDecoding() {
        Log.d(TAG, "startDecoding: 创建解码子线程, 准备开始音频解码");
        final int targetSampleRate = 16000;
        final int targetWindowSize = targetSampleRate * 2; // 32000
        float[] resampledBuffer = new float[targetWindowSize];

        new Thread(() -> {
            long lastPtsMs = -1;  // ✅ 用于检测 pts 是否回退
            isDecoding = true;
            int windowCount = 0;
            boolean sentEos = false;
            boolean outputDone = false;

            MediaExtractor extractor = new MediaExtractor();
            MediaCodec decoder = null;


            try {
                Log.d(TAG, "startDecoding: 打开视频资源...");
                // MediaExtractor.setDataSource(Context, Uri, headers) 同时支持 content://、file:// 和 http(s)://
                // 对网页视频模式必须走这条路径，否则 openAssetFileDescriptor 在 http(s) 上会抛异常
                extractor.setDataSource(context, videoUri, httpHeaders);
                Log.d(TAG, "startDecoding: MediaExtractor 已设置数据源, 开始寻找音频轨道");

                int audioTrackIndex = -1;
                for (int i = 0; i < extractor.getTrackCount(); i++) {
                    MediaFormat format = extractor.getTrackFormat(i);
                    String mime = format.getString(MediaFormat.KEY_MIME);
                    if (mime != null && mime.startsWith("audio/")) {
                        audioTrackIndex = i;
                        break;
                    }
                }

                if (audioTrackIndex < 0) {
                    Log.e(TAG, "startDecoding: 未找到音频轨道！");
                    return;
                }

                Log.d(TAG, "startDecoding: 成功选中音频轨道 " + audioTrackIndex);
                extractor.selectTrack(audioTrackIndex);
                MediaFormat format = extractor.getTrackFormat(audioTrackIndex);

                // ✅ 若有 seek 请求，跳转解码器起始位置
                if (seekToPositionMs >= 0) {
                    extractor.seekTo(seekToPositionMs * 1000L, MediaExtractor.SEEK_TO_CLOSEST_SYNC);
                    Log.d(TAG, "🔄 解码器 seekTo: " + seekToPositionMs + "ms");
                }


                decoder = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME));
                decoder.configure(format, null, null, 0);
                decoder.start();
                Log.d(TAG, "startDecoding: decoder 已启动");

                final int sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE);
                final int windowSize = sampleRate * 4;
                Log.d(TAG, "startDecoding: sampleRate=" + sampleRate + ", windowSize=" + windowSize
                        + ", pipeline=" + (fixedPipeline ? "fixed" : "legacy"));

                // [音频修复] 解码器实际输出的采样率/声道数（INFO_OUTPUT_FORMAT_CHANGED 时更新）
                int[] outFmt = {sampleRate, format.containsKey(MediaFormat.KEY_CHANNEL_COUNT)
                        ? format.getInteger(MediaFormat.KEY_CHANNEL_COUNT) : 2};
                PcmResampler resampler = null;
                long resamplerBaseUs = 0;
                short[] shorts = new short[0];

                ByteBuffer[] inputBuffers = decoder.getInputBuffers();
                ByteBuffer[] outputBuffers = decoder.getOutputBuffers();
                MediaCodec.BufferInfo bufferInfo = new MediaCodec.BufferInfo();
                float[] audioBuffer = new float[windowSize];
                int bufferPos = 0;

                // ✅ [FIX] 新增：记录当前窗口的起始时间戳
                long windowStartPtsMs = -1;

                outerLoop:
                while (isDecoding && !(sentEos && outputDone)) {
                    //Log.d(TAG, "🔁 解码循环进行中...");

                    int inputIndex = decoder.dequeueInputBuffer(10000);
                    if (inputIndex >= 0) {
                        ByteBuffer inputBuffer = inputBuffers[inputIndex];
                        int sampleSize = extractor.readSampleData(inputBuffer, 0);

                        if (sampleSize < 0) {
                            Log.d(TAG, "startDecoding: 读取sampleSize<0, 已到达结尾");
                            decoder.queueInputBuffer(inputIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                            sentEos = true;
                        } else {
                            long presentationTimeUs = extractor.getSampleTime();
                            decoder.queueInputBuffer(inputIndex, 0, sampleSize, presentationTimeUs, 0);
                            extractor.advance();
                            Log.d(TAG, "startDecoding: 向解码器输入 " + sampleSize + " 字节, pts=" + presentationTimeUs);
                        }
                    }

                    int outputIndex = dequeueOutput(decoder, bufferInfo, 10000, outFmt);

                    while (outputIndex >= 0) {
                        long currentPtsMs = bufferInfo.presentationTimeUs / 1000;

                        // 7.24 改进的限速逻辑,防止写入太快
                        long playbackTime = videoPositionProvider.get();

                        // 7.24 计算自上次seek以来的时间
                        long timeSinceSeek = System.currentTimeMillis() - lastSeekTimeInDecoder;

                        // 7.24 根据是否刚seek过来决定允许的最大提前量
                        long maxAdvance;
                        if (timeSinceSeek < SEEK_GRACE_PERIOD && lastSeekTimeInDecoder > 0) {
                            maxAdvance = MAX_ADVANCE_AFTER_SEEK; // seek后2秒内允许提前6秒
                            Log.d(TAG, String.format("🚀 Seek宽限期内，允许提前%d秒 (已过%dms)",
                                    maxAdvance/1000, timeSinceSeek));
                        } else {
                            maxAdvance = MAX_ADVANCE_NORMAL; // 正常情况提前3秒
                        }

                        // 7.24 使用动态的maxAdvance
                        Log.d(TAG, String.format("🔎 写入 PCM：pts=%d, 当前播放=%d, 提前 %.2f 秒, 限制=%.1f秒",
                                currentPtsMs, playbackTime, (currentPtsMs - playbackTime) / 1000.0, maxAdvance / 1000.0));

                        final int availableSamples = pcmBuffer.getBufferedSamples();
                        if (currentPtsMs > playbackTime + maxAdvance) {
                            Log.d(TAG, String.format("⏸️ 解码器限速等待，当前播放时间: %dms，解码帧时间: %dms，超出限制%.1f秒",
                                    playbackTime, currentPtsMs, maxAdvance / 1000.0));
                            Log.d(TAG, "解码器限速等待, 此时 availableSamples=" + availableSamples);
                            try {
                                Thread.sleep(100);
                            } catch (InterruptedException e) {
                                break outerLoop;
                            }
                            continue;
                        }

                        boolean isEosFrame = (bufferInfo.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;

                        Log.d(TAG, "✅ [Decoder] 拿到解码输出帧，pts = " + currentPtsMs + "ms");

                        // ⚠️ 检查 pts 是否异常回退
                        if (lastPtsMs >= 0 && currentPtsMs < lastPtsMs) {
                            Log.w(TAG, "⚠️ 解码输出帧 pts 倒退，当前: " + currentPtsMs + "ms, 上一帧: " + lastPtsMs + "ms，跳过此帧");

                            // 🛑 防止死循环！如果连续出现倒退帧，强制终止解码流程
                            if (currentPtsMs == 0 && currentPtsMs < lastPtsMs) {
                                Log.e(TAG, "🛑 视频播放结束");
                                outputDone = true;
                                sentEos = true;
                                decoder.releaseOutputBuffer(outputIndex, false);
                                break outerLoop;
                            }
                            decoder.releaseOutputBuffer(outputIndex, false);
                            outputIndex = decoder.dequeueOutputBuffer(bufferInfo, 0);
                            continue;
                        }


                        lastPtsMs = currentPtsMs;

                        // ✅ [FIX] 记录窗口起始时间戳（在第一个有效帧时记录）
                        if (windowStartPtsMs < 0) {
                            windowStartPtsMs = currentPtsMs;
                            Log.d(TAG, "📌 [FIX] 记录窗口起始时间戳: " + windowStartPtsMs + "ms");
                        }

                        if (fixedPipeline) {
                            // [音频修复] 整个 buffer 下混 + 重采样，产出的 16kHz 样本连续写入；
                            // 时间戳 = 起头 buffer 的 pts + 样本序号/16000（产出总是 16 的整数倍，正好整毫秒）
                            ByteBuffer ob = decoder.getOutputBuffer(outputIndex);
                            if (ob != null && bufferInfo.size > 0) {
                                ob.position(bufferInfo.offset);
                                ob.limit(bufferInfo.offset + bufferInfo.size);
                                ShortBuffer sb = ob.order(ByteOrder.nativeOrder()).asShortBuffer();
                                int n = sb.remaining();
                                if (shorts.length < n) shorts = new short[n];
                                sb.get(shorts, 0, n);
                                int rate = outFmt[0], ch = Math.max(1, outFmt[1]);
                                long expectUs = resampler == null ? 0
                                        : resamplerBaseUs + resampler.inputFrames() * 1_000_000L / rate;
                                if (resampler == null || resampler.inputRate() != rate
                                        || resampler.channelCount() != ch
                                        || Math.abs(bufferInfo.presentationTimeUs - expectUs) > DISCONTINUITY_US) {
                                    resampler = new PcmResampler(rate, ch);
                                    resamplerBaseUs = bufferInfo.presentationTimeUs;
                                    Log.d(TAG, "[音频修复] 重采样器起头: " + rate + "Hz " + ch + "ch, pts="
                                            + resamplerBaseUs / 1000 + "ms");
                                }
                                long first = resampler.outputCount();
                                float[] out = resampler.process(shorts, 0, n / ch);
                                if (out.length > 0) {
                                    pcmBuffer.write(out, resamplerBaseUs / 1000 + first / 16,
                                            PcmResampler.OUT_RATE);
                                }
                            }
                            if (isEosFrame) {
                                outputDone = true;
                            }
                            decoder.releaseOutputBuffer(outputIndex, false);
                            outputIndex = dequeueOutput(decoder, bufferInfo, 0, outFmt);
                            continue;
                        }

                        // ✅ 正常处理帧数据（原管线）
                        ByteBuffer outputBuffer = outputBuffers[outputIndex];
                        outputBuffer.position(bufferInfo.offset);
                        outputBuffer.limit(bufferInfo.offset + bufferInfo.size);

                        Log.d(TAG, "当前 outputBuffer 剩余字节: " + outputBuffer.remaining());
                        Log.d(TAG, "当前 presentationTimeUs: " + bufferInfo.presentationTimeUs);

                        while (outputBuffer.remaining() >= 2 && bufferPos < windowSize) {
                            short s = outputBuffer.getShort();
                            audioBuffer[bufferPos++] = s / 32768f;
                        }

                        if (bufferPos == windowSize) {
                            Log.d(TAG, "缓冲区满，准备写入 PCM。windowSize=" + windowSize + ", sampleRate=" + sampleRate);

                            // ✅ [FIX] 使用窗口起始时间戳，而不是当前帧的时间戳
                            long segmentStartTimestamp = windowStartPtsMs;

                            if (sampleRate != targetSampleRate) { //将任意采样率的音频重采样到 16kHz
                                float[] originalTime = new float[bufferPos];
                                for (int i = 0; i < bufferPos; i++) {
                                    originalTime[i] = (float) i / sampleRate;
                                }

                                float[] targetTime = new float[targetWindowSize];
                                for (int i = 0; i < targetWindowSize; i++) {
                                    targetTime[i] = (float) i / targetSampleRate;
                                }

                                for (int i = 0, j = 0; i < targetWindowSize; i++) {
                                    while (j < bufferPos - 1 && originalTime[j + 1] < targetTime[i]) {
                                        j++;
                                    }
                                    if (j < bufferPos - 1) {
                                        float t1 = originalTime[j];
                                        float t2 = originalTime[j + 1];
                                        float v1 = audioBuffer[j];
                                        float v2 = audioBuffer[j + 1];
                                        float ratio = (targetTime[i] - t1) / (t2 - t1);
                                        resampledBuffer[i] = v1 + ratio * (v2 - v1);
                                    } else {
                                        resampledBuffer[i] = audioBuffer[j];
                                    }
                                }

                                Log.d(TAG, "📤 [FIX] 使用窗口起始时间戳(ms): " + segmentStartTimestamp);
                                Log.d(TAG, "📤 [write] bufferPos = " + bufferPos + ", sampleRate = " + sampleRate + ", segmentStartTimestamp = " + segmentStartTimestamp);
                                pcmBuffer.write(resampledBuffer, segmentStartTimestamp, targetSampleRate);
                                Log.d(TAG, "✅ 已写入 PCM 缓冲区, 样本数: " + targetWindowSize);
                            } else {
                                Log.d(TAG, "📤 [FIX] 使用窗口起始时间戳(ms): " + segmentStartTimestamp);
                                Log.d(TAG, "📤 [write] bufferPos = " + bufferPos + ", sampleRate = " + sampleRate + ", segmentStartTimestamp = " + segmentStartTimestamp);
                                pcmBuffer.write(audioBuffer, segmentStartTimestamp, sampleRate);
                                Log.d(TAG, "✅ 原始采样率写入 PCM 缓冲区: " + bufferPos + " 样本");
                            }

                            bufferPos = 0;
                            windowStartPtsMs = -1;  // ✅ [FIX] 重置，准备下一个窗口
                            windowCount++;
                        }

                        if (isEosFrame) {
                            Log.d(TAG, "📴 检测到解码器输出 EOS，设置 outputDone = true");
                            outputDone = true;
                        }

                        decoder.releaseOutputBuffer(outputIndex, false);
                        outputIndex = dequeueOutput(decoder, bufferInfo, 0, outFmt);
                    }

                }

                Log.d(TAG, "startDecoding: 解码循环结束, 准备触发 onCompleteListener");
                isDecodingFinished = true;
                if (onCompleteListener != null) {
                    onCompleteListener.run();
                }

                decoder.stop();
                decoder.release();
                extractor.release();
                Log.d(TAG, "startDecoding: MediaCodec 和 MediaExtractor 已释放");

            } catch (IOException e) {
                Log.e(TAG, "音频解码失败: " + e.getMessage());
            }
        }).start();
    }

    /**
     * dequeueOutputBuffer，顺带处理 INFO_OUTPUT_FORMAT_CHANGED：记下解码器实际输出的采样率与声道数
     * （outFmt[0]、outFmt[1]），再取下一个。只在解码线程里调用。
     */
    private static int dequeueOutput(MediaCodec decoder, MediaCodec.BufferInfo info, long timeoutUs,
                                     int[] outFmt) {
        int idx = decoder.dequeueOutputBuffer(info, timeoutUs);
        while (idx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
            MediaFormat f = decoder.getOutputFormat();
            if (f.containsKey(MediaFormat.KEY_SAMPLE_RATE)) outFmt[0] = f.getInteger(MediaFormat.KEY_SAMPLE_RATE);
            if (f.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) outFmt[1] = f.getInteger(MediaFormat.KEY_CHANNEL_COUNT);
            Log.d(TAG, "解码输出格式: " + outFmt[0] + "Hz " + outFmt[1] + "ch");
            idx = decoder.dequeueOutputBuffer(info, 0);
        }
        return idx;
    }

    /**
     * 停止解码线程
     */
    public void stop() {
        Log.d(TAG, "stop: 请求停止解码线程...");
        isDecoding = false;
    }


}
