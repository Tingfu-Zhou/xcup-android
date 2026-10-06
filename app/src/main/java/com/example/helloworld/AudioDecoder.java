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
 * AudioDecoder只负责实时解码并把 PCM 数据送入 PcmCircularBuffer。
 * 每个解码 buffer 交给 {@link PcmResampler}（按声道下混 + 抗混叠重采样到 16kHz），连续写入、不丢样本，
 * 模型窗口就是真正的 [T-2s, T]。原来把交错立体声当单声道、送进模型的是 0.5 倍速音频的处理已删除
 * （见 bench/README.md 的"音频解码修复"）。
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

    /**
     * 解码"轮次"：每次 startDecoding / stop 加一（只在 writeLock 里改）。解码线程只在自己那一轮还是当前轮时工作，
     * 被作废后在下一次检查时退出。原来所有线程共用一个 isDecoding 标志：seekTo 里先 stop() 置 false、紧接着
     * 新线程又置回 true，旧线程（多半停在限速等待里，那里不检查标志）就会接着解码，和新线程一起写缓冲区；
     * 退出页面时 VideoView 已释放、播放位置变成 0，停在限速等待里的线程也永远出不来。
     */
    private volatile int generation = 0;
    /** 写 PCM 与作废旧一轮互斥：stop() 返回之后，旧一轮一个样本也写不进缓冲区。 */
    private final Object writeLock = new Object();

    // 音频分析完成监听器（用于主界面记录音频耗时）
    private Runnable onCompleteListener = null;

    private PcmCircularBuffer pcmBuffer;

    /** 两个相邻解码 buffer 的 pts 与按样本数推算的时间差超过这个，就当作不连续，重采样器重新起头。 */
    private static final long DISCONTINUITY_US = 100_000;

    private volatile boolean isDecodingFinished = false; // 新增：标志位，表示解码是否完成

    private int seekToPositionMs = -1;


    /**
     * 供外部通知 decoder 跳转位置
     */
    // 可接受误差范围（毫秒）
    private static final int SEEK_POSITION_TOLERANCE_MS = 200;

    /**
     * 跳到 ms 重新解码：旧一轮作废 → 清掉缓冲区里的旧音频 → 从新位置开始新一轮。
     * 不再丢弃 1 秒内的第二次 seek：VideoProcessActivity 已经做了 500ms 防抖，丢掉的话解码器会停在上一次的位置
     * （往回拖时缓冲区里一直没有当前位置的音频）。
     */
    public void seekTo(int ms) {
        // 7.24新增：记录seek时间，供解码线程使用
        lastSeekTimeInDecoder = System.currentTimeMillis();

        Log.d(TAG, "🔄 AudioDecoder.seekTo() called, new time: " + ms + "ms");

        stop();
        // 旧一轮作废之后再清：从这里起它不会再写缓冲区，旧位置的音频不会混进新的窗口
        pcmBuffer.reset();
        this.seekToPositionMs = ms;  // 直接seek到当前位置，让限速机制自然控制提前量。
        startDecoding();
        Log.d(TAG, "🚀 seekTo() -> startDecoding(), seekToPositionMs = " + seekToPositionMs);

    }


    public AudioDecoder(Context context, Uri videoUri, PcmCircularBuffer buffer, Supplier<Integer> videoPositionProvider) {
        this(context, videoUri, null, buffer, videoPositionProvider);
    }

    public AudioDecoder(Context context, Uri videoUri, Map<String, String> httpHeaders,
                        PcmCircularBuffer buffer, Supplier<Integer> videoPositionProvider) {
        this.context = context;
        this.videoUri = videoUri;
        this.httpHeaders = httpHeaders;
        this.pcmBuffer = buffer;
        this.videoPositionProvider = videoPositionProvider;
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
        final int run;
        synchronized (writeLock) {
            run = ++generation;          // 新的一轮；之前那一轮（如果还在跑）随之作废
        }
        final int startMs = seekToPositionMs;
        Log.d(TAG, "startDecoding: 创建解码子线程（第 " + run + " 轮）, 准备开始音频解码");

        new Thread(() -> {
            long lastPtsMs = -1;  // ✅ 用于检测 pts 是否回退
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
                if (startMs >= 0) {
                    extractor.seekTo(startMs * 1000L, MediaExtractor.SEEK_TO_CLOSEST_SYNC);
                    Log.d(TAG, "🔄 解码器 seekTo: " + startMs + "ms");
                }


                decoder = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME));
                decoder.configure(format, null, null, 0);
                decoder.start();
                Log.d(TAG, "startDecoding: decoder 已启动");

                final int sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE);
                Log.d(TAG, "startDecoding: sampleRate=" + sampleRate);

                // 解码器实际输出的采样率/声道数（INFO_OUTPUT_FORMAT_CHANGED 时更新）
                int[] outFmt = {sampleRate, format.containsKey(MediaFormat.KEY_CHANNEL_COUNT)
                        ? format.getInteger(MediaFormat.KEY_CHANNEL_COUNT) : 2};
                PcmResampler resampler = null;
                long resamplerBaseUs = 0;
                short[] shorts = new short[0];

                ByteBuffer[] inputBuffers = decoder.getInputBuffers();
                MediaCodec.BufferInfo bufferInfo = new MediaCodec.BufferInfo();

                outerLoop:
                while (generation == run && !(sentEos && outputDone)) {
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
                        if (generation != run) {
                            break outerLoop;   // 已被 seek / stop 作废（多半正停在下面的限速等待里）
                        }
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

                        // 整个 buffer 下混 + 重采样，产出的 16kHz 样本连续写入；
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
                                Log.d(TAG, "重采样器起头: " + rate + "Hz " + ch + "ch, pts="
                                        + resamplerBaseUs / 1000 + "ms");
                            }
                            long first = resampler.outputCount();
                            float[] out = resampler.process(shorts, 0, n / ch);
                            if (out.length > 0) {
                                synchronized (writeLock) {
                                    if (generation != run) {
                                        break outerLoop;   // 作废之后一个样本也不再写
                                    }
                                    pcmBuffer.write(out, resamplerBaseUs / 1000 + first / 16,
                                            PcmResampler.OUT_RATE);
                                }
                            }
                        }

                        if (isEosFrame) {
                            Log.d(TAG, "📴 检测到解码器输出 EOS，设置 outputDone = true");
                            outputDone = true;
                        }

                        decoder.releaseOutputBuffer(outputIndex, false);
                        outputIndex = dequeueOutput(decoder, bufferInfo, 0, outFmt);
                    }

                }

                if (generation == run) {
                    Log.d(TAG, "startDecoding: 解码循环结束, 准备触发 onCompleteListener");
                    isDecodingFinished = true;
                    if (onCompleteListener != null) {
                        onCompleteListener.run();
                    }
                } else {
                    Log.d(TAG, "startDecoding: 第 " + run + " 轮已被 seek / stop 作废，退出");
                }

            } catch (IOException e) {
                Log.e(TAG, "音频解码失败: " + e.getMessage());
            } finally {
                // 不管怎么退出都释放解码器（被作废的一轮也不再占着它）
                if (decoder != null) {
                    try {
                        decoder.stop();
                    } catch (RuntimeException ignored) {
                        // 没 start 成功或已出错的解码器 stop() 会抛异常，照样 release
                    }
                    decoder.release();
                }
                extractor.release();
                Log.d(TAG, "startDecoding: MediaCodec 和 MediaExtractor 已释放");
            }
        }, "AudioDecoder-" + run).start();
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
     * 停止解码线程：当前这一轮作废。返回之后它不会再往缓冲区写任何数据，
     * 并在下一次检查时（最多一个限速等待周期，约 100ms）退出、释放解码器。
     */
    public void stop() {
        Log.d(TAG, "stop: 请求停止解码线程...");
        synchronized (writeLock) {
            generation++;
        }
    }


}
