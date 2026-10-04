package com.example.helloworld;

import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.util.Log;
import android.widget.MediaController;
import android.widget.TextView;
import android.widget.VideoView;
import android.graphics.Bitmap;

import androidx.appcompat.app.AppCompatActivity;

import android.widget.Toast;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicBoolean;

import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.view.TextureView;
import android.view.View;
import android.widget.ImageButton;

import android.widget.FrameLayout;
import android.view.Gravity;
import android.content.res.Configuration;

import androidx.annotation.OptIn;
import androidx.media3.common.PlaybackException;
import androidx.media3.common.Player;
import androidx.media3.common.VideoSize;
import androidx.media3.common.util.UnstableApi;
import androidx.media3.datasource.HttpDataSource;
import androidx.media3.exoplayer.ExoPlayer;
import androidx.media3.ui.PlayerView;

@OptIn(markerClass = UnstableApi.class)
public class VideoProcessActivity extends AppCompatActivity {

    private static final String TAG = "VideoProcessActivity";
    private VideoView videoView;
    private PlayerView playerView;                  // [WebVideo] 仅网页模式可见
    private ExoPlayerEngine exoPlayerEngine;        // [WebVideo] 仅网页模式创建
    private long webVideoStartPositionMs = 0;       // [WebVideo] 从 WebView 带过来的起播时间
    private boolean exoFirstReadyHandled = false;   // [WebVideo] STATE_READY 只处理第一次
    private TextView tvOverlay;
    private TextView tvVideoAction;
    private TextView tvAudioAction;
    private VideoFrameExtractor videoFrameExtractor;
    private VideoClassifierHelper videoClassifier;
    //private BluetoothHelper bluetoothHelper;
    private Handler playStateHandler;
    private Runnable playStateChecker;

    private boolean isVideoCompleted = false;

    // ✅ 音频推理相关变量
    private AudioDecoder audioDecoder;
    private AudioInferenceHelper audioHelper;

    private FrameLayout videoContainer;  // UI新增
    private int originalWidth = 0;       // UI新增
    private int originalHeight = 0;      // UI新增

    // 使用原子引用来安全地在线程间共享结果
    private final AtomicReference<String> latestAudioAction = new AtomicReference<>("");
    private final AtomicReference<String> latestVideoAction = new AtomicReference<>("");
    private final AtomicReference<Float> latestAudioConfidence = new AtomicReference<>(0f);
    private final AtomicReference<Float> latestVideoConfidence = new AtomicReference<>(0f);
    // 显示蓝牙发送的动作
    private final AtomicReference<String> latestBluetoothAction = new AtomicReference<>("");
    // 添加时间戳记录
    private final AtomicReference<Long> latestVideoTimestamp = new AtomicReference<>(0L);
    private final AtomicReference<Long> latestAudioTimestamp = new AtomicReference<>(0L);

    // [证据融合] 见 EvidenceFusion。assets/fusion_config.json 的 mode 决定它是否驱动蓝牙：
    // legacy=不运行；shadow=每 tick 计算并写进评测 CSV，蓝牙仍由原融合驱动；live=由它驱动蓝牙。
    // 配置缺失或解析失败时回退到原融合。
    private static final String FUSION_CONFIG_ASSET = "fusion_config.json";
    private EvidenceFusion evidenceFusion;          // null = 不运行
    private boolean evidenceLive = false;           // true = 证据融合驱动蓝牙
    private String fusionModeTag = "legacy";        // 写进评测参数行，例如 shadow/provisional-20261004
    // 模型结果快照（概率 + 产出时刻）作为一个对象原子发布，证据融合不会读到"新概率配旧时间戳"
    private final AtomicReference<EvidenceFusion.Snapshot> latestVideoSnap = new AtomicReference<>(null);
    private final AtomicReference<EvidenceFusion.Snapshot> latestAudioSnap = new AtomicReference<>(null);

    // 7.19 新增：时间窗口平滑相关变量
    // [延迟优化] 窗口从 10 收缩到 6：融合循环现为 500ms 一次，6 条记录约覆盖 3 秒。
    // 稳定性保护统一交给 updateBluetoothState 的状态机（最短持续/发送间隔/停转确认），
    // 平滑窗口只负责压制单次抖动，不再叠加长时间惯性。
    private static final int SMOOTH_WINDOW_SIZE = 6;
    // 动作在窗口内至少出现的次数（音视频各计一次），低于该次数不参与投票
    private static final int SMOOTH_MIN_COUNT = 3;
    // [快通道] 音视频同一 tick 给出相同动作且双双高置信度时，绕过投票直接输出该动作。
    // 停转方向的安全性由蓝牙状态机兜底（do->Noise 仍需稳定 1000ms + 最短持续 2000ms）。
    private static final float FUSION_FASTPATH_VIDEO_CONF = 0.70f;
    private static final float FUSION_FASTPATH_AUDIO_CONF = 0.70f;
    // [启动仲裁] 实测音频模型准确度高于视频模型，因此视频不对启动拥有否决权，
    // 只在“视频剧情证据明显强于音频 do 证据”时短暂延迟启动（有硬上限，绝不永久阻塞）。
    // 冲突判定：videoAction==Noise 且 videoConf >= VIDEO_PLOT_CONFLICT_CONF
    //           且 (videoConf - audioDoConf) >= FUSION_CONFLICT_MARGIN
    private static final float VIDEO_PLOT_CONFLICT_CONF = 0.80f;
    private static final float FUSION_CONFLICT_MARGIN = 0.15f;
    // 冲突延迟上限：连续冲突超过该 tick 数后仍放行，避免视频误判把音频永久锁死
    private static final int CONFLICT_MAX_HOLD_TICKS = 4;
    // 强音频直通：音频 do 置信度达到该值时无需视频背书，直接允许启动
    private static final float AUDIO_STRONG_START_CONF = 0.75f;
    // 弱证据兜底：音频 do 但置信度不高、且窗口内无视频 do 证据时，需连续这么多 tick 才启动
    private static final int AUDIO_ONLY_START_TICKS = 2;
    private int audioOnlyDoStreak = 0;  // 连续纯音频 do 的融合 tick 计数（仅主线程融合循环访问）
    private int conflictHoldTicks = 0;  // 连续音视频强冲突的 tick 计数（仅主线程融合循环访问）
    private final LinkedList<ActionRecord> actionHistory = new LinkedList<>();
    private final Object historyLock = new Object(); // 用于同步访问actionHistory

    // 7.19 新增：内部类定义动作记录
    private static class ActionRecord {
        String videoAction;
        String audioAction;
        float videoConfidence;
        float audioConfidence;
        long timestamp;

        ActionRecord(String vAction, String aAction, float vConf, float aConf) {
            this.videoAction = vAction != null ? vAction : "";
            this.audioAction = aAction != null ? aAction : "";
            this.videoConfidence = vConf;
            this.audioConfidence = aConf;
            this.timestamp = System.currentTimeMillis();
        }
    }

    // 蓝牙发送状态管理器变量部分
    private static final long BLUETOOTH_MIN_DURATION = 2000; // 蓝牙动作最小持续时间2秒
    private static final long BLUETOOTH_SEND_INTERVAL = 1600; // 蓝牙发送间隔1600ms
    private long lastBluetoothSendTime = 0;
    private String currentBluetoothState = "";
    private long currentStateStartTime = 0;
    private String pendingBluetoothState = "";
    private long pendingStateStartTime = 0;

    // ★ 新增：暂停时挂起的蓝牙动作与档位
    private String suspendedBluetoothState = "";
    private int suspendedLevel = 0;

    // NEW: 频率档位确认与节流相关
    private int currentLevel = 1;                 // 已生效档位（0..10）
    private long currentLevelSinceMs = 0;         // 当前档位生效起始时间
    private Integer pendingLevel = null;          // 待确认档位
    private long pendingLevelSinceMs = 0;         // 待确认计时

    // NEW: 参数（可调）
    private static final int LEVEL_STABLE_MS   = 0;   // 新档位需稳定的最短时间
    private static final int LEVEL_MIN_DUR_MS  = 0;  // 生效档位的最小驻留

    // 最近一次“已发送”档位（用于判断是否需要重发同一动作以更新档位）
    private int lastSentLevel = 0;                  // NEW: 记录上次发送出去的档位

    // ✅ 音频分析计时
    private long audioStartTime = 0;
    private boolean isAudioCompleted = false;
    private PcmCircularBuffer pcmBuffer;
    private Handler seekHandler = new Handler(Looper.getMainLooper());
    private Runnable pendingSeekRunnable;
    private boolean isFullscreen = false;
    private ImageButton btnFullscreen;
    private ImageButton btnBack;

    // 线程控制
    private final AtomicBoolean isAnalysisPaused = new AtomicBoolean(false);
    private final AtomicBoolean shouldStop = new AtomicBoolean(false);

    // 后台线程和Handler
    private HandlerThread videoThread;
    private Handler videoHandler;
    private HandlerThread audioThread;
    private Handler audioHandler;
    private Handler mainHandler;

    // 线程间同步
    private final Object videoLock = new Object();
    private final Object audioLock = new Object();

    // 帧缓存环形缓冲：最近 3 秒、间隔 250ms 采样的 12 帧
    private final ArrayDeque<Bitmap> frameWindow = new ArrayDeque<>();
    private static final int FRAME_WINDOW_SIZE = VideoClassifierHelper.NUM_FRAMES;

    // 视频推理步长：每 500ms 推理一次（推理约 110ms + 抽帧 43ms，2Hz 下 CPU 占用可接受）
    private long lastVideoInferenceTime = 0;
    private static final long VIDEO_INFERENCE_INTERVAL = 500;

    // [分组概率决策] 下游只关心“转/不转”，因此在 oral+sex 合并后的概率上做判定，
    // 而不是先 argmax 单类再过阈值——避免转场期 P(oral)=0.4/P(sex)=0.4 这类
    // “总证据充分但单类不过线”的结果被误判为 unclear。
    // P(do) = P(oral)+P(sex) >= DO 阈值 -> "do"（转）
    // P(plot/noise) >= PLOT/NOISE 阈值 -> "Noise"（不转）
    // 两者都不满足 -> unclear（""，置信度 0）
    private static final float VIDEO_DO_PROB_THRESHOLD = 0.65f;
    private static final float VIDEO_PLOT_PROB_THRESHOLD = 0.60f;
    private static final float AUDIO_DO_PROB_THRESHOLD = 0.55f;
    private static final float AUDIO_NOISE_PROB_THRESHOLD = 0.60f;

    // 主线程融合循环间隔（毫秒）
    // [延迟优化] 1000 -> 500：融合本身耗时约 40-60ms，加快采样让动作切换更快被捕捉；
    // 蓝牙实际发送频率仍由 BLUETOOTH_SEND_INTERVAL / 状态机节流，不会过载。
    private static final long MAIN_FUSION_INTERVAL = 500;
    // 视频线程间隔（帧缓存间隔 250ms）
    private static final long VIDEO_LOOP_INTERVAL_MS = 250;
    //音频线程间隔
    private static final long AUDIO_LOOP_TICK_MS = 1000;

    // [12.30] Loudness 档位估计器（音频节律输出）
    private final AudioLoudnessLevelEstimator loudnessEstimator = new AudioLoudnessLevelEstimator(16000);

    // [12.30] 音频”响度档位”结果（线程安全共享）
    private final java.util.concurrent.atomic.AtomicInteger latestAudioLoudLevel =
            new java.util.concurrent.atomic.AtomicInteger(0);
    private final java.util.concurrent.atomic.AtomicReference<Float> latestAudioLoudConf =
            new java.util.concurrent.atomic.AtomicReference<>(0f);
    private final java.util.concurrent.atomic.AtomicLong latestAudioLoudTsMs =
            new java.util.concurrent.atomic.AtomicLong(0L);
    private final java.util.concurrent.atomic.AtomicBoolean latestAudioLoudValid =
            new java.util.concurrent.atomic.AtomicBoolean(false);

    // [WebVideo] 网页视频模式相关
    // 网页视频模式下没有 seek，没有 MediaController，分析的开/停由音频 VAD（参考在线模式）决定
    private boolean isWebVideoMode = false;
    // VAD 状态
    private boolean webVadActive = false;
    private long webVadLastActiveMs = 0;
    private int webVadConsecutiveSilentTicks = 0;
    // VAD 参数（音频循环每秒一次，5 个 silent tick ≈ 5 秒静音才暂停）
    private static final int WEB_VAD_MIN_SILENT_TICKS = 5;
    private static final long WEB_VAD_SILENCE_MS = 5000;
    private static final float WEB_VAD_ACTIVATE_RMS_DB = -55f;
    private static final float WEB_VAD_ACTIVATE_PEAK_DB = -42f;
    private static final float WEB_VAD_KEEP_RMS_DB = -60f;
    private static final float WEB_VAD_KEEP_PEAK_DB = -48f;
    private long lastWebVadLogMs = 0;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Intent intent = getIntent();
        super.onCreate(savedInstanceState);
        Log.d(TAG, "onCreate: 正在初始化布局...");
        setContentView(R.layout.activity_video_process);

        // 初始化UI组件
        videoView = findViewById(R.id.videoView);
        playerView = findViewById(R.id.playerView);          // [WebVideo]
        videoContainer = findViewById(R.id.videoContainer);  // 🔴 新增这行
        tvOverlay = findViewById(R.id.tvOverlay);
        tvVideoAction = findViewById(R.id.tvVideoAction);
        tvAudioAction = findViewById(R.id.tvAudioAction);
        btnFullscreen = findViewById(R.id.btnFullscreen);

        // 返回主页面的初始化和点击监听
        btnBack = findViewById(R.id.btnBack);
        btnBack.setOnClickListener(v -> finish());

        // 正式版隐藏悬浮窗调试UI
        // tvOverlay.setVisibility(View.GONE);
        // tvVideoAction.setVisibility(View.GONE);
        // tvAudioAction.setVisibility(View.GONE);

        videoClassifier = new VideoClassifierHelper(this);
        Log.d(TAG, "onCreate: VideoClassifierHelper (MobileNetV3Small) 初始化完成");

        // [WebVideo] 解析网页视频模式标志和 HTTP headers
        isWebVideoMode = intent.getBooleanExtra(WebVideoActivity.EXTRA_IS_WEB_VIDEO, false);
        Map<String, String> httpHeaders = null;
        if (isWebVideoMode) {
            httpHeaders = new HashMap<>();
            String referer = intent.getStringExtra(WebVideoActivity.EXTRA_HTTP_REFERER);
            String ua = intent.getStringExtra(WebVideoActivity.EXTRA_HTTP_UA);
            String cookie = intent.getStringExtra(WebVideoActivity.EXTRA_HTTP_COOKIE);
            if (referer != null && !referer.isEmpty()) httpHeaders.put("Referer", referer);
            if (ua != null && !ua.isEmpty()) httpHeaders.put("User-Agent", ua);
            if (cookie != null && !cookie.isEmpty()) httpHeaders.put("Cookie", cookie);
            webVideoStartPositionMs = Math.max(0L,
                    intent.getLongExtra(WebVideoActivity.EXTRA_START_POSITION_MS, 0L));
            Log.d(TAG, "onCreate: 网页视频模式启用，headers=" + httpHeaders.keySet()
                    + ", startMs=" + webVideoStartPositionMs);
        }

        Uri videoUri = intent.getData();
        if (videoUri == null) {
            Log.e(TAG, "onCreate: 没有有效的视频 URI，退出！");
            finish();
            return;
        }

        // [证据融合] 读配置；必须在 BenchLogger.init 之前，参数行里要记下融合模式
        initEvidenceFusion();

        // [评测] 把每个融合 tick dump 成 CSV，供 bench/tools/metrics.py 算指标。
        // debug 包自动开启；只读取不干预，关闭时全是空操作。
        BenchLogger.get().init(this, videoUri, benchParamsLine());

        // 通用：音频推理 + PCM 缓冲。两种模式都需要。
        Log.d(TAG, "onCreate: 初始化 AudioInferenceHelper + PcmCircularBuffer...");
        audioHelper = new AudioInferenceHelper(this);
        pcmBuffer = new PcmCircularBuffer(16000, 20); // 采样率 16kHz，最多缓冲 20 秒

        if (isWebVideoMode) {
            // ============== 网页视频模式：ExoPlayer + PlayerView ==============
            // VideoView / VideoFrameExtractor / AudioDecoder 全部不启用，避免对
            // HLS 走 MediaExtractor 那条已知不通的路径。
            videoView.setVisibility(View.GONE);
            playerView.setVisibility(View.VISIBLE);
            // 网页视频默认 analysis 暂停，由音频 VAD 驱动 resume/pause
            isAnalysisPaused.set(true);
            setupExoPlayerForWebMode(videoUri, httpHeaders);
        } else {
            // ============== 离线模式：保持原 VideoView + MediaExtractor 路径 ==============
            videoView.setVideoURI(videoUri);
            Log.d(TAG, "onCreate: 设置视频 URI: " + videoUri + " (offline)");

            try {
                videoFrameExtractor = new VideoFrameExtractor(this, videoUri, null);
                Log.d(TAG, "onCreate: VideoFrameExtractor 初始化成功");
            } catch (IOException e) {
                Log.e(TAG, "onCreate: VideoFrameExtractor 初始化失败 uri=" + videoUri, e);
                Toast.makeText(this, "视频解析失败：" + e.getMessage(), Toast.LENGTH_LONG).show();
                finish();
                return;
            }

            videoView.setMediaController(new MediaController(this));
            videoView.requestFocus();

            videoView.setOnCompletionListener(mp -> {
                isVideoCompleted = true;
                pauseAnalysis();
                Log.d(TAG, "视频播放完成！暂停同步分析，等待用户操作");
            });

            videoView.setOnPreparedListener(mp -> {
                originalWidth = mp.getVideoWidth();
                originalHeight = mp.getVideoHeight();
                Log.d(TAG, "视频原始尺寸: " + originalWidth + "x" + originalHeight);
                adjustVideoSize(false);
                mp.setOnSeekCompleteListener(seekMp -> handleSeekComplete());
                videoView.start();
                startMultiThreadAnalysis();
            });

            videoView.setOnErrorListener((mp, what, extra) -> {
                Log.e(TAG, "视频播放出错 what:" + what + " extra:" + extra
                        + " uri=" + videoUri);
                String hint;
                switch (extra) {
                    case -1004: hint = "IO 错误（403/网络中断）"; break;
                    case -1007: hint = "畸形/不支持的流"; break;
                    case -1010: hint = "不支持的格式"; break;
                    case -110:  hint = "请求超时"; break;
                    default:    hint = "未知错误";
                }
                Toast.makeText(this,
                        "VideoView 播放失败 (what=" + what + ", extra=" + extra + ", " + hint + ")",
                        Toast.LENGTH_LONG).show();
                return true;
            });

            audioDecoder = new AudioDecoder(this, videoUri, null, pcmBuffer,
                    () -> videoView.getCurrentPosition());
            audioDecoder.setOnCompleteListener(() -> {
                isAudioCompleted = true;
                Log.d(TAG, "音频解码完成");
            });
            audioDecoder.startDecoding();
            Log.d(TAG, "onCreate: audioDecoder 已启动解码线程");
        }

        // 全屏按钮
        btnFullscreen.setOnClickListener(v -> {
            if (!isFullscreen) {
                enterFullscreen();
            } else {
                exitFullscreen();
            }
        });

        // 播放状态监听（仅离线模式）
        // 网页视频模式下分析的开/停由音频 VAD 驱动，不靠 VideoView.isPlaying()
        if (!isWebVideoMode) {
            playStateHandler = new Handler(Looper.getMainLooper());
            playStateChecker = new Runnable() {
                boolean lastPlayingState = true;

                @Override
                public void run() {
                    boolean isPlayingNow = videoView.isPlaying();
                    if (isPlayingNow != lastPlayingState) {
                        if (isPlayingNow) {
                            resumeAnalysis();
                        } else {
                            pauseAnalysis();
                        }
                        lastPlayingState = isPlayingNow;
                    }
                    playStateHandler.postDelayed(this, 200);
                }
            };
            playStateHandler.post(playStateChecker);
        }

        // 初始化主线程Handler
        mainHandler = new Handler(Looper.getMainLooper());
    }

    /**
     * 启动多线程分析
     */
    private void startMultiThreadAnalysis() {
        Log.d(TAG, "启动多线程分析系统...");

        // 创建视频分析线程
        videoThread = new HandlerThread("VideoAnalysisThread");
        videoThread.start();
        videoHandler = new Handler(videoThread.getLooper());

        // 创建音频分析线程
        audioThread = new HandlerThread("AudioAnalysisThread");
        audioThread.start();
        audioHandler = new Handler(audioThread.getLooper());

        // 启动视频分析循环
        startVideoAnalysisLoop();

        // 启动音频分析循环
        startAudioAnalysisLoop();

        // 启动主线程融合循环
        startMainFusionLoop();
    }

    /**
     * 视频分析循环 - 在独立线程中运行。
     * 每 250ms 抽取并缓存一帧；缓存窗口满 12 帧（3 秒）后，每 1000ms 执行一次推理。
     */
    private void startVideoAnalysisLoop() {
        Runnable videoRunnable = new Runnable() {
            @Override
            public void run() {
                if (shouldStop.get()) {
                    Log.d(TAG, "[视频线程] 收到停止信号，退出循环");
                    return;
                }

                if (isAnalysisPaused.get()) {
                    videoHandler.postDelayed(this, VIDEO_LOOP_INTERVAL_MS);
                    return;
                }

                long t0 = System.currentTimeMillis();

                try {
                    long currentMs = getCurrentPlaybackPosMs();
                    final long currentMsFinal = currentMs;

                    // 在主线程抽帧：离线走 MediaCodec+EGLRenderer，网页模式直接读 TextureView
                    mainHandler.post(() -> {
                        try {
                            long tExtractStart = System.currentTimeMillis();
                            Bitmap frame;
                            if (isWebVideoMode) {
                                frame = captureWebVideoFrameBitmap();
                            } else {
                                frame = videoFrameExtractor.getFrameAt(currentMsFinal * 1000);
                            }
                            Log.d(TAG, "[计时] 📸 抽帧耗时: "
                                    + (System.currentTimeMillis() - tExtractStart) + " ms");

                            if (frame != null) {
                                videoHandler.post(() -> cacheFrameAndInfer(frame));
                            } else {
                                Log.w(TAG, "[主线程] [同步分析] 未能抽取到视频帧");
                            }
                        } catch (Exception e) {
                            Log.e(TAG, "[主线程] 抽帧异常：", e);
                        }
                    });

                } catch (Exception e) {
                    Log.e(TAG, "[视频线程] 异常：", e);
                }

                long elapsed = System.currentTimeMillis() - t0;
                long nextDelay = Math.max(0, VIDEO_LOOP_INTERVAL_MS - elapsed);
                videoHandler.postDelayed(this, nextDelay);
            }
        };

        videoHandler.post(videoRunnable);
    }

    /**
     * 缓存一帧到滑动窗口；窗口满 12 帧且距上次推理 ≥ 1000ms 时执行一次推理。
     */
    private void cacheFrameAndInfer(Bitmap frame) {
        synchronized (videoLock) {
            // 缩放到模型输入尺寸后缓存
            Bitmap scaled = Bitmap.createScaledBitmap(
                    frame, VideoClassifierHelper.INPUT_SIZE, VideoClassifierHelper.INPUT_SIZE, true);
            frameWindow.add(scaled);
            while (frameWindow.size() > FRAME_WINDOW_SIZE) {
                frameWindow.poll();
            }

            // 滑动窗口未满 3 秒，暂不推理
            if (frameWindow.size() < FRAME_WINDOW_SIZE) {
                Log.d(TAG, "[视频线程] 帧窗口预热中 (" + frameWindow.size()
                        + "/" + FRAME_WINDOW_SIZE + ")");
                return;
            }

            // 推理步长 1000ms
            long now = System.currentTimeMillis();
            if (now - lastVideoInferenceTime < VIDEO_INFERENCE_INTERVAL) {
                return;
            }
            lastVideoInferenceTime = now;

            Bitmap[] frames = frameWindow.toArray(new Bitmap[0]);
            long tInferStart = System.currentTimeMillis();
            VideoClassifierHelper.Result result = videoClassifier.predict(frames);
            Log.d(TAG, "[计时] 🧠 MobileNetV3Small 推理耗时: "
                    + (System.currentTimeMillis() - tInferStart) + " ms");

            applyVideoResult(result);
        }
    }

    /**
     * 将三分类结果映射为视频动作并写入共享结果（分组概率决策）。
     * P(do)=P(oral)+P(sex) >= 阈值 -> "do"（转）；P(plot) >= 阈值 -> "Noise"（不转）；
     * 其余 -> unclear -> ""（置信度 0）。置信度取分组后的概率。
     */
    private void applyVideoResult(VideoClassifierHelper.Result result) {
        String actionClass;
        float confidence;

        if (result == null || result.index < 0 || result.probs == null || result.probs.length < 3) {
            actionClass = "";
            confidence = 0f;
            Log.w(TAG, "[同步分析] [视频线程] 推理结果无效，判为 unclear");
        } else {
            float pPlot = result.probs[0];
            float pDo = result.probs[1] + result.probs[2]; // oral + sex 合并为“转”的总证据

            if (pDo >= VIDEO_DO_PROB_THRESHOLD) {
                actionClass = "do";      // 转
                confidence = pDo;
            } else if (pPlot >= VIDEO_PLOT_PROB_THRESHOLD) {
                actionClass = "Noise";   // 不转
                confidence = pPlot;
            } else {
                actionClass = "";
                confidence = 0f;
                Log.d(TAG, String.format("[同步分析] [视频线程] P(do)=%.2f / P(plot)=%.2f 均未过阈值，判为 unclear",
                        pDo, pPlot));
            }
        }

        // [证据融合] 原始概率 + 产出时刻打包成快照原子发布（评测 CSV 的 v_p_* 也取自它）；
        // 推理失败时 probs 为 null，不给证据
        long resultTs = System.currentTimeMillis();
        boolean probsOk = result != null && result.index >= 0
                && result.probs != null && result.probs.length >= 3;
        latestVideoSnap.set(new EvidenceFusion.Snapshot(probsOk ? result.probs.clone() : null, resultTs));

        // 原子更新结果
        latestVideoAction.set(actionClass);
        latestVideoConfidence.set(confidence);
        latestVideoTimestamp.set(resultTs);

        final String displayText = actionClass.isEmpty()
                ? "V: unclear"
                : String.format("V: %s (p=%.2f)", actionClass, confidence);
        mainHandler.post(() -> tvVideoAction.setText(displayText));
        Log.d(TAG, "[同步分析] [视频线程] " + displayText
                + (result != null ? " probs=" + Arrays.toString(result.probs) : ""));
    }

    /**
     * 音频分析循环 - 在独立线程中运行
     * 动作类别：label 0 对应着做爱，label 1 对应着口交，label 2 对应着杂音。
     */
    private long lastAudioInferenceTime = 0; // 记录上次音频推理的时间
    private static final long AUDIO_INFERENCE_INTERVAL = 1000; // 音频推理间隔1秒

    private void startAudioAnalysisLoop() {
        Runnable audioRunnable = new Runnable() {
            @Override
            public void run() {
                if (shouldStop.get()) {
                    Log.d(TAG, "[音频线程] 收到停止信号，退出循环");
                    return;
                }

                // [WebVideo] 网页视频模式：先跑 VAD（不论当前是否暂停），由 VAD 决定要不要 resume/pause
                if (isWebVideoMode) {
                    runWebVideoVadTick();
                }

                if (isAnalysisPaused.get()) {
                    audioHandler.postDelayed(this, 1000);
                    return;
                }
                long t0 = System.currentTimeMillis();

                long currentMs = getCurrentPlaybackPosMs();
                long currentSystemTime = System.currentTimeMillis();

                // 控制音频推理频率, 每1秒执行一次
                if ((currentSystemTime - lastAudioInferenceTime) >= AUDIO_INFERENCE_INTERVAL) {
                    synchronized (audioLock) {
                        // [延迟优化] 死区 4000 -> 2000：模型窗口只需 [T-2s,T] 共 2 秒数据，
                        // AudioDecoder 本身保持约 3 秒解码提前量；且 readWindowRelaxed 在样本
                        // 不足 90% 时会返回 null 自我保护，等满 2 秒即可开始尝试。
                        if ((currentMs - audioStartTime) >= 2000) {
                            // 读取音频数据
                            float[] audioSegment = pcmBuffer.readWindowRelaxed(currentMs, 32000);

                            // **************[12.30] 音频 Loudness → 档位分析（替代频率估计）**************
                            float[] last1s = pcmBuffer.readWindowRelaxed(currentMs, 8000); // 0.5秒, 16 kHz/s
                            if (last1s != null && last1s.length > 0) {
                                loudnessEstimator.push(last1s);
                                AudioLoudnessLevelEstimator.Result lr = loudnessEstimator.estimate(System.currentTimeMillis());

                                Log.d(TAG, String.format("[响度档位] valid:%s, level:%d, conf:%.2f, db:%.1f",
                                        lr.valid, lr.level, lr.confidence, lr.db));

                                latestAudioLoudLevel.set(lr.level);
                                latestAudioLoudConf.set(lr.confidence);
                                latestAudioLoudTsMs.set(lr.timestampMs);
                                latestAudioLoudValid.set(lr.valid);
                            }
                            //********************************************************************

                            if (audioSegment != null) {
                                // 音频推理
                                // ⏱️ 音频推理计时
                                long tAudioInferStart = System.currentTimeMillis();
                                AudioInferenceHelper.AudioInferenceResult result = audioHelper.predict(audioSegment);
                                long tAudioInferEnd = System.currentTimeMillis();
                                Log.d(TAG, "[计时] 🔊 音频推理耗时: " + (tAudioInferEnd - tAudioInferStart) + " ms");

                                // 音频模型类别: 0=sex(做爱), 1=oral(口交), 2=noise(杂音)
                                // [分组概率决策] P(do)=P(sex)+P(oral) >= 阈值 -> "do"(转);
                                // P(noise) >= 阈值 -> "Noise"(不转); 其余 -> ""(unclear)
                                String action;
                                float confidence;
                                if (result.index < 0 || result.probs.length < 3) {
                                    action = "";
                                    confidence = 0f;
                                    Log.w(TAG, "[音频线程] [同步分析] 推理结果无效，判为 unclear");
                                } else {
                                    float pDo = result.probs[0] + result.probs[1]; // sex + oral 合并
                                    float pNoise = result.probs[2];

                                    if (pDo >= AUDIO_DO_PROB_THRESHOLD) {
                                        action = "do";      // 转
                                        confidence = pDo;
                                    } else if (pNoise >= AUDIO_NOISE_PROB_THRESHOLD) {
                                        action = "Noise";   // 不转
                                        confidence = pNoise;
                                    } else {
                                        action = "";
                                        confidence = 0f;
                                        Log.d(TAG, String.format("[音频线程] [同步分析] P(do)=%.2f / P(noise)=%.2f 均未过阈值，判为 unclear",
                                                pDo, pNoise));
                                    }
                                }

                                // [证据融合] 原始概率 + 产出时刻打包成快照原子发布（评测 CSV 的 a_p_* 也取自它）
                                long resultTs = System.currentTimeMillis();
                                boolean probsOk = result.index >= 0 && result.probs.length >= 3;
                                latestAudioSnap.set(new EvidenceFusion.Snapshot(
                                        probsOk ? result.probs.clone() : null, resultTs));

                                // 原子更新结果
                                latestAudioAction.set(action);
                                latestAudioConfidence.set(confidence);
                                latestAudioTimestamp.set(resultTs);

                                // UI更新
                                final String displayText = action.isEmpty()
                                        ? "A: unclear"
                                        : String.format("A: %s (p=%.2f)", action, confidence);
                                mainHandler.post(() -> tvAudioAction.setText(displayText));
                                Log.d(TAG, "[同步分析] [音频线程] " + displayText);
                                // 更新上次音频推理时间
                                lastAudioInferenceTime = currentSystemTime;
                            } else {
                                Log.w(TAG, "[音频线程] [同步分析] 未能读取到音频数据");
                            }
                        } else {
                            Log.w(TAG, "[同步分析] PCM 缓冲启动中，等待解码器填充...");
                        }
                    }
                } else {
                    // 不执行音频推理时的日志
                    Log.d(TAG, "[音频] 距上次推理时间不足1秒，跳过本次音频推理");
                }

                // 计算本轮耗时并做动态延迟
                long elapsed = System.currentTimeMillis() - t0;
                Log.d(TAG, "[音频线程] 本轮调度耗时: " + elapsed + "ms");
                long nextDelay = Math.max(0, AUDIO_LOOP_TICK_MS - elapsed);
                // 继续下一轮
                audioHandler.postDelayed(this, nextDelay);
            }
        };

        audioHandler.post(audioRunnable);
    }

    /**
     * 主线程融合循环 - 负责结果融合、UI更新和蓝牙发送
     */
    private void startMainFusionLoop() {
        Runnable fusionRunnable = new Runnable() {
            @Override
            public void run() {
                if (shouldStop.get()) {
                    return;
                }

                // 检查BLE是否被本地按键暂停
                if (BLEManager.globalManager != null && BLEManager.globalManager.isPausedByLocal()) {
                    Log.d(TAG, "[融合线程] BLE被本地按键暂停，跳过蓝牙发送");
                    // 仍然进行分析但不发送蓝牙命令
                }

                // 检查是否暂停
                if (isAnalysisPaused.get()) {
                    Log.d(TAG, "[融合线程] 当前处于暂停状态，跳过本轮融合");
                    mainHandler.postDelayed(this, MAIN_FUSION_INTERVAL);
                    return;
                }
                long t0 = System.currentTimeMillis();

                // 读取最新的分析结果（原子操作，线程安全）
                String videoAction = latestVideoAction.get();
                String audioAction = latestAudioAction.get();
                float videoConf = latestVideoConfidence.get();
                float audioConf = latestAudioConfidence.get();
                long videoTime = latestVideoTimestamp.get();
                long audioTime = latestAudioTimestamp.get();
                // [证据融合] 模型结果快照（与上面的动作串最多差一个推理周期，各自内部一致）
                EvidenceFusion.Snapshot vSnap = latestVideoSnap.get();
                EvidenceFusion.Snapshot aSnap = latestAudioSnap.get();

                // 音频节奏
                boolean audioFreqValid = latestAudioLoudValid.get();
                float audioFreq = (float) latestAudioLoudLevel.get();    // [MOD] 用 level 伪装成“freq输入”
                float audioFreqConf = latestAudioLoudConf.get();
                long audioFreqTs = latestAudioLoudTsMs.get();

                // 计算结果的新鲜度（毫秒）
                long currentTime = System.currentTimeMillis();
                long videoAge = videoTime > 0 ? currentTime - videoTime : Long.MAX_VALUE;
                long audioAge = audioTime > 0 ? currentTime - audioTime : Long.MAX_VALUE;
                long audioFreqAge = audioFreqTs > 0 ? currentTime - audioFreqTs : Long.MAX_VALUE;

                // 过滤超过2秒的过期数据
                final long MAX_AGE = 2000; // 2秒

                // 如果音频频节奏结果过期，清空它
                if (audioFreqAge > MAX_AGE || !audioFreqValid) {
                    audioFreq = Float.NaN;
                    audioFreqConf = 0f;
                    if (audioFreqAge > MAX_AGE) {
                        Log.d(TAG, "[融合] 音频节奏结果过期（" + audioFreqAge + "ms），已忽略");
                    } else {
                        Log.d(TAG, "[融合] 音频节奏结果无效（audioFreqValid=false），已忽略");
                    }
                }

                // 如果视频结果过期，清空它
                if (videoAge > MAX_AGE) {
                    videoAction = "";
                    videoConf = 0f;
                    Log.d(TAG, "[融合] 视频结果过期（" + videoAge + "ms），已忽略");
                }

                // 如果音频结果过期，清空它
                if (audioAge > MAX_AGE) {
                    audioAction = "";
                    audioConf = 0f;
                    Log.d(TAG, "[融合] 音频结果过期（" + audioAge + "ms），已忽略");
                }

                // 注：oral/sex 的归并已在各分析线程内完成（applyVideoResult / 音频循环
                // 的分组概率决策只输出 "do"/"Noise"/""），此处无需再做 oral->do 归一化。

                // [证据融合] 每 tick 都算：shadow 模式只写进评测 CSV；live 模式由它驱动蓝牙，
                // 原来的投票窗与启动仲裁不再参与决策（蓝牙状态机照旧）
                EvidenceFusion.Output ev = evidenceFusion == null ? null
                        : evidenceFusion.update(vSnap, aSnap, currentTime);

                String finalAction;
                final String smoothedAction;
                if (evidenceLive && ev != null) {
                    finalAction = ev.action;
                    smoothedAction = ev.action;
                } else {
                    // 7.19 修改：使用平滑融合替代原有的简单融合逻辑
                    finalAction = smoothedFusion(videoAction, audioAction, videoConf, audioConf);
                    // [评测] 留存平滑级输出，dump 时才能把融合延迟和仲裁延迟分开看
                    smoothedAction = finalAction;

                    // [启动仲裁] 音视频强冲突时短暂延迟启动（有上限）；音频足够自信则直接启动
                    finalAction = applyDoStartGate(finalAction, videoAction, videoConf, audioAction, audioConf);
                }
                // 临时采用音频节律作为最终节律
                // 将“最终节律计算”封装为独立方法，便于后续替换为音视频融合节律
                int finalFreq = computeFinalFreq(audioFreq, audioFreqConf);

                // [修改] 检查BLE暂停状态，如果未暂停才发送
                if (!finalAction.isEmpty()) {
                    if (BLEManager.globalManager == null || !BLEManager.globalManager.isPausedByLocal()) {
                        updateBluetoothState(finalAction,finalFreq);
                        Log.d(TAG, String.format("[融合] finalAction: %s (V:%dms前, A:%dms前)",
                                finalAction, videoAge, audioAge));
                    } else {
                        Log.d(TAG, "[融合] BLE暂停中，跳过动作: " + finalAction);
                    }
                }

                // [评测] dump 本 tick。必须放在 updateBluetoothState 之后，才能带上本轮
                // 是否真的发了蓝牙指令——切换延迟就是靠这一列定位的。
                if (BenchLogger.get().isEnabled()) {
                    // v_age/a_age 取快照的年龄：概率和"它是哪一刻出的"必须来自同一个结果，
                    // 离线回放（fusion_lab.py）靠 wall_ms - age 还原结果时刻做去重
                    BenchLogger.get().tick(
                            getCurrentPlaybackPosMs(), currentTime,
                            videoAction, videoConf,
                            vSnap == null ? Long.MAX_VALUE : currentTime - vSnap.tsMs,
                            vSnap == null ? null : vSnap.probs,
                            audioAction, audioConf,
                            aSnap == null ? Long.MAX_VALUE : currentTime - aSnap.tsMs,
                            aSnap == null ? null : aSnap.probs,
                            Float.isNaN(audioFreq) ? -1 : Math.round(audioFreq),
                            audioFreqConf, audioFreqValid, audioFreqAge,
                            smoothedAction, finalAction, finalFreq,
                            currentBluetoothState, currentLevel, ev);
                }

                // 更新UI, 显示蓝牙实际发送的动作
                String bluetoothAction = latestBluetoothAction.get();
                if (!bluetoothAction.isEmpty()) {
                    tvOverlay.setText("蓝牙: " + bluetoothAction + "节奏：" + currentLevel);
                } else {
                    tvOverlay.setText("蓝牙发送等待识别...");
                }

                // 继续下一轮
                long elapsed = System.currentTimeMillis() - t0;
                Log.d(TAG, "[融合线程] 本轮调度耗时: " + elapsed + "ms");
                long nextDelay = Math.max(0, MAIN_FUSION_INTERVAL - elapsed);
                mainHandler.postDelayed(this, nextDelay);
            }
        };

        mainHandler.post(fusionRunnable);
    }

    /**
     * 计算最终节律档位（0..8）。
     *
     * <p>策略：使用音频响度档位，并做“置信度阈值 + 方向性门控（涨档更严格，降档更宽松）”。</p>
     */
    private int computeFinalFreq(float audioFreq, float audioFreqConf) {
        int finalFreq = clampLevelFromLoudness(audioFreq); // audioFreq 实际是 loudness level(float)
        Log.d(TAG, "[音频节律] 得到音频档位: " + audioFreq + " 置信度: " + audioFreqConf + " 档位: " + finalFreq);

        // === 12.12: 方向性置信度门控（涨档更严格，降档更宽松） ===
        {
            float conf = audioFreqConf;      // 当前这帧的置信度

            // 三个可调参数
            final float CONF_IGNORE = 0.10f;  // 极低置信度：整体忽略本次节律
            final float CONF_UP     = 0.10f;  // 涨档所需置信度
            final float CONF_DOWN   = 0.10f;  // 降档所需置信度

            int curLevel       = currentLevel;  // 当前已生效档位（0..10）
            int candidateLevel = finalFreq;     // 本次根据 audioFreq 映射出来的档位

            // 1) 极低置信度：直接清空本次节律，维持 currentLevel
            if (Float.isNaN(audioFreq) || conf < CONF_IGNORE) {
                Log.d(TAG, String.format(
                        "[音频节律] conf=%.2f < CONF_IGNORE=%.2f，本次节律整体忽略，freq=NaN，沿用 currentLevel=%d",
                        conf, CONF_IGNORE, curLevel));
                finalFreq = curLevel;  // 不给 updateBluetoothState 提供变档机会
            } else {
                // 2) 根据档位变动方向应用不同门槛
                if (candidateLevel > curLevel && conf < CONF_UP) {
                    // 尝试“涨档”但置信度不足 → 不允许涨档
                    Log.d(TAG, String.format(
                            "[音频节律] 尝试涨档 %d→%d 但 conf=%.2f < CONF_UP=%.2f，本次不生效，沿用 currentLevel=%d",
                            curLevel, candidateLevel, conf, CONF_UP, curLevel));
                    finalFreq = curLevel;
                } else if (candidateLevel < curLevel && conf < CONF_DOWN) {
                    // 尝试“降档”但置信度也太低 → 不允许降档（可视需要放宽）
                    Log.d(TAG, String.format(
                            "[音频节律] 尝试降档 %d→%d 但 conf=%.2f < CONF_DOWN=%.2f，本次不生效，沿用 currentLevel=%d",
                            curLevel, candidateLevel, conf, CONF_DOWN, curLevel));
                    finalFreq = curLevel;
                }
                // candidateLevel == curLevel 时无需处理
            }
        }

        return finalFreq;
    }

    // [ADD] audioFreq 实际承载的是 loudness level（float），这里做 0..8 钳制
    private static int clampLevelFromLoudness(final float levelLike) {
        if (Float.isNaN(levelLike)) return 0;
        int lv = Math.round(levelLike);
        if (lv < 0) lv = 0;
        if (lv > 8) lv = 8;
        return lv;
    }


    // 7.19 新增：时间窗口平滑融合方法
    private String smoothedFusion(String videoAction, String audioAction, float videoConf, float audioConf) {
        synchronized (historyLock) {
            // 添加当前记录到历史
            actionHistory.add(new ActionRecord(videoAction, audioAction, videoConf, audioConf));

            // 保持窗口大小
            while (actionHistory.size() > SMOOTH_WINDOW_SIZE) {
                actionHistory.poll();
            }

            // [快通道] 音视频本 tick 一致且双双高置信度 -> 绕过投票直接输出。
            // 平滑窗口的意义是压制单模态抖动；两个模态同时高置信度一致时没有抖动嫌疑，
            // 不必再等窗口攒票。停转方向仍有蓝牙状态机兜底（稳定 1000ms + 最短持续 2000ms）。
            if (!videoAction.isEmpty() && videoAction.equals(audioAction)
                    && videoConf >= FUSION_FASTPATH_VIDEO_CONF
                    && audioConf >= FUSION_FASTPATH_AUDIO_CONF) {
                Log.d(TAG, String.format("[平滑融合] 快通道命中: %s (V:%.2f, A:%.2f)",
                        videoAction, videoConf, audioConf));
                return videoAction;
            }

            // 如果历史记录太少，使用原始逻辑
            if (actionHistory.size() < 3) {
                return selectBestAction(videoAction, audioAction, videoConf, audioConf);
            }

            // 统计各动作的加权得分
            Map<String, Float> actionScores = new HashMap<>();
            Map<String, Integer> actionCounts = new HashMap<>();

            // 给最近的记录更高的权重
            int index = 0;
            for (ActionRecord record : actionHistory) {
                float weight = (float)(index + 1) / actionHistory.size(); // 时间权重，越新权重越高

                // 处理视频动作
                if (!record.videoAction.isEmpty() && !record.videoAction.equals("Background")) {
                    // 包括Noise在内的所有动作都参与评分
                    String videoKey = record.videoAction;
                    float score = actionScores.getOrDefault(videoKey, 0f);
                    score += record.videoConfidence * weight * 0.8f; // 视频权重0.8
                    actionScores.put(videoKey, score);

                    int count = actionCounts.getOrDefault(videoKey, 0);
                    actionCounts.put(videoKey, count + 1);
                }

                // 处理音频动作
                if (!record.audioAction.isEmpty()) {
                    // 包括Noise在内的所有动作都参与评分
                    String audioKey = record.audioAction;
                    float score = actionScores.getOrDefault(audioKey, 0f);
                    score += record.audioConfidence * weight * 1.2f; // 音频权重1.2
                    actionScores.put(audioKey, score);

                    int count = actionCounts.getOrDefault(audioKey, 0);
                    actionCounts.put(audioKey, count + 1);
                }

                index++;
            }

            // 选择得分最高的动作
            String bestAction = "";
            float bestScore = 0;

            for (Map.Entry<String, Float> entry : actionScores.entrySet()) {
                String action = entry.getKey();
                float score = entry.getValue();
                int count = actionCounts.getOrDefault(action, 0);

                // 需要至少出现 SMOOTH_MIN_COUNT 次才考虑（避免偶然噪声）
                if (count >= SMOOTH_MIN_COUNT && score > bestScore) {
                    bestScore = score;
                    bestAction = action;
                }
            }

            // 如果没有找到合适的动作，使用最新的高置信度结果
            if (bestAction.isEmpty()) {
                ActionRecord latest = actionHistory.getLast();
                bestAction = selectBestAction(latest.videoAction, latest.audioAction,
                        latest.videoConfidence, latest.audioConfidence);
            }

            // 记录平滑结果
            Log.d(TAG, String.format("[平滑融合] 窗口大小:%d, 选择动作:%s (得分:%.2f)",
                    actionHistory.size(), bestAction, bestScore));

            return bestAction;
        }
    }

    /**
     * [启动仲裁] 对融合结果为 "do" 的<b>启动</b>过程做冲突仲裁；仅影响“从非 do 状态启动 do”，
     * 已处于 do 状态时（维持阶段）音频可以单独维持，不做任何限制。
     *
     * <p>设计前提：实测音频模型准确度高于视频模型，因此视频<b>不</b>拥有否决权，只能在
     * 与音频强烈冲突时把启动短暂推迟，且推迟有硬上限。规则按序：</p>
     * <ol>
     *   <li><b>强冲突延迟</b>：视频判剧情且置信度 ≥ {@link #VIDEO_PLOT_CONFLICT_CONF}，
     *       同时比音频的 do 置信度高出 {@link #FUSION_CONFLICT_MARGIN} 以上 -> 本 tick 不启动；
     *       但连续冲突超过 {@link #CONFLICT_MAX_HOLD_TICKS} 个 tick 后强制放行，
     *       避免视频持续误判（暗光/特写/非常规机位）把正确的音频永久锁死。</li>
     *   <li><b>强音频直通</b>：音频 do 置信度 ≥ {@link #AUDIO_STRONG_START_CONF} -> 立即放行，
     *       不需要视频背书。</li>
     *   <li><b>视频证据放行</b>：平滑窗口内存在任一条视频 "do" -> 放行。</li>
     *   <li><b>弱证据兜底</b>：以上都不满足（音频 do 但置信度不高、视频也没有证据）时，
     *       要求连续 {@link #AUDIO_ONLY_START_TICKS} 个 tick 才启动。</li>
     * </ol>
     *
     * <p>被拦下时返回 ""（本 tick 不驱动蓝牙状态机，维持现状），不主动发送 Noise。
     * 最坏情况下的启动延迟为 CONFLICT_MAX_HOLD_TICKS + AUDIO_ONLY_START_TICKS = 6 tick（约 3 秒）。</p>
     */
    private String applyDoStartGate(String finalAction, String videoAction, float videoConf,
                                    String audioAction, float audioConf) {
        if (!"do".equals(finalAction)) {
            audioOnlyDoStreak = 0;
            conflictHoldTicks = 0;
            return finalAction;
        }
        // 维持阶段：当前蓝牙状态已是 do，音频单独维持即可
        if (isSexAction(currentBluetoothState)) {
            audioOnlyDoStreak = 0;
            conflictHoldTicks = 0;
            return finalAction;
        }

        // 音频本 tick 支持 do 时的置信度；音频不支持 do 则记为 0
        float audioDoConf = "do".equals(audioAction) ? audioConf : 0f;

        // 规则 1：强冲突延迟（有上限，绝不永久阻塞）
        boolean strongConflict = "Noise".equals(videoAction)
                && videoConf >= VIDEO_PLOT_CONFLICT_CONF
                && (videoConf - audioDoConf) >= FUSION_CONFLICT_MARGIN;
        if (strongConflict) {
            conflictHoldTicks++;
            if (conflictHoldTicks <= CONFLICT_MAX_HOLD_TICKS) {
                Log.d(TAG, String.format(
                        "[启动仲裁] 音视频强冲突(V:Noise %.2f vs A:do %.2f)，延迟启动 (%d/%d tick)",
                        videoConf, audioDoConf, conflictHoldTicks, CONFLICT_MAX_HOLD_TICKS));
                return "";
            }
            // 冲突持续超过上限：视频可能在系统性误判，按“音频更可信”放行
            Log.d(TAG, String.format(
                    "[启动仲裁] 强冲突已持续 %d tick 超过上限，按音频优先放行启动", conflictHoldTicks));
        } else {
            conflictHoldTicks = 0;
        }

        // 规则 2：强音频直通
        if (audioDoConf >= AUDIO_STRONG_START_CONF) {
            audioOnlyDoStreak = 0;
            Log.d(TAG, String.format("[启动仲裁] 音频 do 置信度 %.2f 达直通门槛，放行启动", audioDoConf));
            return finalAction;
        }

        // 规则 3：窗口内有视频 do 证据 -> 放行
        synchronized (historyLock) {
            for (ActionRecord record : actionHistory) {
                if ("do".equals(record.videoAction)) {
                    audioOnlyDoStreak = 0;
                    return finalAction;
                }
            }
        }

        // 规则 4：弱证据兜底，需要连续多个 tick 支持
        audioOnlyDoStreak++;
        if (audioOnlyDoStreak >= AUDIO_ONLY_START_TICKS) {
            Log.d(TAG, "[启动仲裁] 弱证据 do 已连续 " + audioOnlyDoStreak + " tick，放行启动");
            return finalAction;
        }
        Log.d(TAG, String.format("[启动仲裁] 弱证据 do (%d/%d tick)，暂不启动",
                audioOnlyDoStreak, AUDIO_ONLY_START_TICKS));
        return "";
    }

    // 简单的动作选择逻辑（用于历史记录不足时）
    private String selectBestAction(String videoAction, String audioAction, float videoConf, float audioConf) {
        // 使用置信度来决定，而不是排除Noise
        // 音频优先策略
        if (!audioAction.isEmpty()) {
            // 如果音频是有效动作（非Noise）或音频置信度很高，使用音频
            if (!audioAction.equals("Noise") || audioConf > 0.7f) {
                return audioAction;
            }
        }
        // 视频作为备选
        if (!videoAction.isEmpty() && !videoAction.equals("Background")) {
            // 如果视频是有效动作（非Noise）或视频置信度很高，使用视频
            if (!videoAction.equals("Noise") || videoConf > 0.7f) {
                return videoAction;
            }
        }
        // 如果音视频都是Noise，返回Noise（而不是空字符串）
        if ("Noise".equals(audioAction) || "Noise".equals(videoAction)) {
            return "Noise";
        }

        return "";
    }


    // 蓝牙发送状态管理器
    private void updateBluetoothState(String newAction, int finalFreq /* 0..8 */) {
        long currentTime = System.currentTimeMillis();

        // =====================[ 档位确认：立即生效版 ]=====================
        // 只要 newAction 是 do/oral，就允许处理档位。
        // 不再依赖 currentStateSupportsSpeed()，避免从 Noise -> do 时 currentBluetoothState 还不是 do，导致档位不同步。
        boolean supportsLevel = isSexAction(newAction);

        if (supportsLevel) {
            int latestLevel = finalFreq;

            // 安全钳制到 0..8
            if (latestLevel < 0) latestLevel = 0;
            if (latestLevel > 8) latestLevel = 8;

            // 迟滞：只有跨过 currentLevel ± 1 才处理
            boolean worthHandling =
                    latestLevel >= upThreshold(currentLevel) ||
                            latestLevel <= downThreshold(currentLevel);

            if (worthHandling) {
                // 关键修复：
                // LEVEL_STABLE_MS = 0 时，第一次看到新档位就立即生效，不再等下一轮融合循环。
                if (LEVEL_STABLE_MS <= 0 &&
                        (currentTime - currentLevelSinceMs >= LEVEL_MIN_DUR_MS)) {

                    if (latestLevel != currentLevel) {
                        Log.d(TAG, String.format(
                                "[档位确认] 立即生效: %d -> %d",
                                currentLevel, latestLevel
                        ));
                    }

                    currentLevel = latestLevel;
                    currentLevelSinceMs = currentTime;
                    pendingLevel = null;
                    pendingLevelSinceMs = 0;

                } else {
                    // 如果以后你把 LEVEL_STABLE_MS 改成 >0，则走这个短稳确认逻辑
                    if (pendingLevel == null || !pendingLevel.equals(latestLevel)) {
                        pendingLevel = latestLevel;
                        pendingLevelSinceMs = currentTime;

                        Log.d(TAG, String.format(
                                "[档位确认] 新 pendingLevel=%d，开始等待稳定",
                                latestLevel
                        ));

                    } else {
                        long dwell = currentTime - pendingLevelSinceMs;
                        boolean stableOk = dwell >= LEVEL_STABLE_MS;
                        boolean minDurOk = currentTime - currentLevelSinceMs >= LEVEL_MIN_DUR_MS;

                        if (stableOk && minDurOk) {
                            if (pendingLevel != currentLevel) {
                                Log.d(TAG, String.format(
                                        "[档位确认] pending 生效: %d -> %d, dwell=%dms",
                                        currentLevel, pendingLevel, dwell
                                ));
                            }

                            currentLevel = pendingLevel;
                            currentLevelSinceMs = currentTime;
                            pendingLevel = null;
                            pendingLevelSinceMs = 0;
                        }
                    }
                }
            } else {
                // 没有跨过迟滞门槛，清空 pending，避免旧 pending 干扰后续判断
                pendingLevel = null;
                pendingLevelSinceMs = 0;
            }
        } else {
            // Noise 或其他不支持变速的动作，不处理档位
            pendingLevel = null;
            pendingLevelSinceMs = 0;
        }
        // =====================[ 档位确认结束 ]=====================


        // =====================[ 动作待确认 ]=====================
        if (!newAction.equals(pendingBluetoothState)) {
            pendingBluetoothState = newAction;
            pendingStateStartTime = currentTime;
            Log.d(TAG, String.format("[蓝牙] 检测到新动作: %s, 等待确认...", newAction));
        }

        // 动作稳定确认时间：
        // 默认 0ms；如果从 do/oral 切到 Noise，则延长到 1000ms，避免动作中突然停。
        int actionStableMs = 0;
        if (isSexAction(currentBluetoothState) && "Noise".equals(newAction)) {
            actionStableMs = 1000;
        }

        boolean actionStableOk =
                pendingBluetoothState.equals(newAction) &&
                        (currentTime - pendingStateStartTime) >= actionStableMs;

        if (!actionStableOk) {
            Log.d(TAG, String.format(
                    "[蓝牙] 动作尚未稳定: pending=%s, new=%s, 已稳定=%dms, 需要=%dms",
                    pendingBluetoothState,
                    newAction,
                    currentTime - pendingStateStartTime,
                    actionStableMs
            ));
            return;
        }


        // =====================[ 情况 A：动作发生变化 ]=====================
        if (!pendingBluetoothState.equals(currentBluetoothState)) {

            boolean minDurationOk =
                    currentBluetoothState.isEmpty() ||
                            (currentTime - currentStateStartTime) >= BLUETOOTH_MIN_DURATION;

            if (!minDurationOk) {
                long remainingTime = BLUETOOTH_MIN_DURATION - (currentTime - currentStateStartTime);
                Log.d(TAG, String.format(
                        "[蓝牙] 当前动作 %s 需继续保持 %dms，暂不切换到 %s",
                        currentBluetoothState,
                        remainingTime,
                        pendingBluetoothState
                ));
                return;
            }

            boolean gapOk = (currentTime - lastBluetoothSendTime) >= BLUETOOTH_SEND_INTERVAL;

            if (!gapOk) {
                Log.d(TAG, String.format(
                        "[蓝牙] 动作切换等待发送间隔: gap=%dms, need=%dms",
                        currentTime - lastBluetoothSendTime,
                        BLUETOOTH_SEND_INTERVAL
                ));
                return;
            }

            int levelToSend;
            if (isSexAction(pendingBluetoothState)) {
                // 动作切换到 do/oral 时，发送已确认的 currentLevel
                levelToSend = currentLevel;
            } else {
                // 切到 Noise 时，档位归 0
                levelToSend = 0;
            }

            Log.i(TAG, String.format(
                    "[蓝牙] [同步分析] ✅ 发送指令: action=%s, level=%d, 已稳定=%dms",
                    pendingBluetoothState,
                    levelToSend,
                    currentTime - pendingStateStartTime
            ));

            latestBluetoothAction.set(pendingBluetoothState);
            BenchLogger.get().noteSend(pendingBluetoothState, levelToSend);

            if (BLEManager.globalManager != null && BLEManager.globalManager.isConnected()) {
                BLEManager.globalManager.sendAction(pendingBluetoothState, levelToSend);
                Log.i(TAG, "[蓝牙] [同步分析] ✅ 已通过BLE发送指令(动作切换/含档位)");
                Log.d(TAG, "[音频节律] ✅ 发送节律：" + levelToSend);
            } else {
                Log.w(TAG, "[蓝牙] BLE未连接，仅更新UI显示");
            }

            currentBluetoothState = pendingBluetoothState;
            currentStateStartTime = currentTime;
            lastBluetoothSendTime = currentTime;

            // 关键同步：
            // 保证 currentLevel / lastSentLevel / 实际发送档位一致。
            if (isSexAction(currentBluetoothState)) {
                currentLevel = levelToSend;
                currentLevelSinceMs = currentTime;
                lastSentLevel = levelToSend;
            } else {
                lastSentLevel = 0;
            }

            return;
        }


        // =====================[ 情况 B：动作未变，但档位变化 ]=====================
        // 动作不变，例如 do -> do，但档位从 8 降到 4。
        // 这种情况需要重发同一个动作，携带新档位。
        boolean levelChanged = supportsLevel && (currentLevel != lastSentLevel);
        boolean gapOk = (currentTime - lastBluetoothSendTime) >= BLUETOOTH_SEND_INTERVAL;

        Log.d(TAG, String.format(
                "[档位发送判断] action=%s, supportsLevel=%s, currentLevel=%d, lastSentLevel=%d, levelChanged=%s, gapOk=%s, gap=%dms",
                currentBluetoothState,
                supportsLevel,
                currentLevel,
                lastSentLevel,
                levelChanged,
                gapOk,
                currentTime - lastBluetoothSendTime
        ));

        if (levelChanged && gapOk) {
            int levelToSend = currentLevel;

            Log.i(TAG, String.format(
                    "[蓝牙] 同动作更新档位：%s -> level=%d",
                    currentBluetoothState,
                    levelToSend
            ));

            BenchLogger.get().noteSend(currentBluetoothState, levelToSend);

            if (BLEManager.globalManager != null && BLEManager.globalManager.isConnected()) {
                BLEManager.globalManager.sendAction(currentBluetoothState, levelToSend);
                Log.i(TAG, "[蓝牙] [同步分析] ✅ 已通过BLE发送指令(同动作/更新档位)");
                Log.d(TAG, "[音频节律] ✅ 发送节律：" + levelToSend);
            } else {
                Log.w(TAG, "[蓝牙] BLE未连接，无法更新档位（同动作）");
            }

            lastSentLevel = levelToSend;
            lastBluetoothSendTime = currentTime;
        }
    }


    // （可选）门控：当前动作是否支持“变速”（你若已有类似函数，可直接替换）
    // 根据你的动作命名规则自行实现判断逻辑
    private boolean currentStateSupportsSpeed() {
        return currentBluetoothState.startsWith("do")
                || currentBluetoothState.startsWith("oral");
    }

    // （可选）门控：newAction 是否属于“做爱大类”（你若已有类似函数，可直接替换）
    private boolean isSexAction(String action) {
        return action != null && (
                action.startsWith("do") ||
                        action.startsWith("oral")
        );
    }

    // 迟滞门限辅助
    private int upThreshold(int cur)   { return Math.min(10, cur + 1); }
    private int downThreshold(int cur) { return Math.max(0,  cur - 1); }

    /**
     * [证据融合] 读 assets/fusion_config.json。缺文件、格式错、参数非法都回退到原融合，绝不影响正常使用。
     * 配置由 bench/tools/fusion_lab.py fit 生成。
     */
    private void initEvidenceFusion() {
        evidenceFusion = null;
        evidenceLive = false;
        fusionModeTag = "legacy";
        try (InputStream in = getAssets().open(FUSION_CONFIG_ASSET)) {
            ByteArrayOutputStream buf = new ByteArrayOutputStream();
            byte[] b = new byte[4096];
            int n;
            while ((n = in.read(b)) > 0) buf.write(b, 0, n);
            EvidenceFusion.Config cfg = EvidenceFusion.Config.fromJson(
                    new String(buf.toByteArray(), StandardCharsets.UTF_8));
            fusionModeTag = (cfg.mode + "/" + cfg.id).replaceAll("\\s+", "_");
            if (EvidenceFusion.MODE_LEGACY.equals(cfg.mode)) {
                Log.i(TAG, "[证据融合] 配置 mode=legacy，不运行");
                return;
            }
            evidenceFusion = new EvidenceFusion(cfg);
            evidenceLive = EvidenceFusion.MODE_LIVE.equals(cfg.mode);
            Log.i(TAG, "[证据融合] 已加载 " + cfg.id + "，mode=" + cfg.mode
                    + (cfg.provisional ? "（临时配置，未经回放验证）" : ""));
        } catch (Exception e) {
            fusionModeTag = "legacy(配置无效)";
            Log.e(TAG, "[证据融合] 读取 " + FUSION_CONFIG_ASSET + " 失败，回退到原融合", e);
        }
    }

    /** [证据融合] seek 时调用（只在主线程）：清空累积证据和 seek 前的模型结果快照。 */
    private void resetEvidenceFusion() {
        latestVideoSnap.set(null);
        latestAudioSnap.set(null);
        if (evidenceFusion != null) {
            evidenceFusion.reset();
        }
    }

    /**
     * [评测] 把当前这组可调参数拼成一行，写进 run CSV 的文件头。
     *
     * <p>没有这一行，事后就分不清某份 dump 是哪组参数跑出来的，“每改一个参数都能量化
     * 对比”也就无从谈起。加了新参数记得同步加到这里。
     */
    private String benchParamsLine() {
        return String.format(Locale.US,
                "VIDEO_DO=%.2f VIDEO_PLOT=%.2f AUDIO_DO=%.2f AUDIO_NOISE=%.2f "
                        + "SMOOTH_WIN=%d SMOOTH_MIN=%d FASTPATH_V=%.2f FASTPATH_A=%.2f "
                        + "PLOT_CONFLICT=%.2f CONFLICT_MARGIN=%.2f CONFLICT_HOLD=%d "
                        + "AUDIO_STRONG=%.2f AUDIO_ONLY_TICKS=%d "
                        + "BLE_MIN_DUR=%d BLE_GAP=%d LEVEL_STABLE=%d LEVEL_MIN_DUR=%d "
                        + "FUSION_TICK=%d VIDEO_INFER=%d AUDIO_INFER=%d FUSION=%s",
                VIDEO_DO_PROB_THRESHOLD, VIDEO_PLOT_PROB_THRESHOLD,
                AUDIO_DO_PROB_THRESHOLD, AUDIO_NOISE_PROB_THRESHOLD,
                SMOOTH_WINDOW_SIZE, SMOOTH_MIN_COUNT,
                FUSION_FASTPATH_VIDEO_CONF, FUSION_FASTPATH_AUDIO_CONF,
                VIDEO_PLOT_CONFLICT_CONF, FUSION_CONFLICT_MARGIN, CONFLICT_MAX_HOLD_TICKS,
                AUDIO_STRONG_START_CONF, AUDIO_ONLY_START_TICKS,
                BLUETOOTH_MIN_DURATION, BLUETOOTH_SEND_INTERVAL,
                LEVEL_STABLE_MS, LEVEL_MIN_DUR_MS,
                MAIN_FUSION_INTERVAL, VIDEO_INFERENCE_INTERVAL, AUDIO_INFERENCE_INTERVAL,
                fusionModeTag);
    }



    /**
     * 处理视频seek完成
     */
    private void handleSeekComplete() {
        int currentMs = videoView.getCurrentPosition();
        Log.w(TAG, "[同步分析] 🟡 onSeekComplete(): 用户完成拖动，跳转到 " + currentMs + "ms，准备延迟处理 seek");

        // 取消之前挂起的 seek 请求（防抖）
        if (pendingSeekRunnable != null) {
            seekHandler.removeCallbacks(pendingSeekRunnable);
        }
        // 延迟 500ms 执行最终 seek 逻辑（只处理最后一次）
        pendingSeekRunnable = () -> {
            Log.i(TAG, "执行最终seek处理，时间点：" + currentMs);

            // 清空帧缓存和 pcm 缓冲，避免读取到“前一个时间点”的旧数据
            synchronized (videoLock) {
                frameWindow.clear();
                lastVideoInferenceTime = 0;
            }

            synchronized (audioLock) {
                pcmBuffer.reset();
            }

            // 7.19新增：清空动作历史记录
            synchronized (historyLock) {
                actionHistory.clear();
                Log.d(TAG, "[平滑融合] 清空历史记录（因为seek操作）");
            }
            // [启动仲裁] seek 后重置启动计数
            audioOnlyDoStreak = 0;
            conflictHoldTicks = 0;
            // [证据融合] seek 前的证据和结果快照都不能算进新位置
            resetEvidenceFusion();

            // 重置提取器, 通知视频帧抽取器：从当前时间开始解码
            videoFrameExtractor.seekTo(currentMs * 1000);
            // decoder 负责重新 seek 解码
            audioDecoder.seekTo(currentMs);
            Log.d(TAG, "🎯 audioDecoder.seekTo() 调用，当前视频播放时间: " + currentMs);
            audioStartTime = currentMs;

            // 如果之前已经播到结尾，清零标记并恢复分析
            if (isVideoCompleted) {
                isVideoCompleted = false;
                // 只有当前已经在播放，才立刻恢复分析；否则继续保持暂停
                if (videoView.isPlaying()) {
                    resumeAnalysis();
                }
            }
            // 清空蓝牙控制动作缓存
            currentBluetoothState = "";
            pendingBluetoothState = "";
            latestBluetoothAction.set("");
            currentStateStartTime = 0;
            pendingStateStartTime = 0;
            // 重置档位确认状态
            currentLevel = 1;
            currentLevelSinceMs = 0;
            pendingLevel = null;
            pendingLevelSinceMs = 0;

            // ★ 新增：清空挂起状态（seek后旧动作已无意义）
            suspendedBluetoothState = "";
            suspendedLevel = 0;

            // [12.30] reset loudness estimator
            loudnessEstimator.reset();
            // [12.30] 清空 loudness 输出
            latestAudioLoudLevel.set(0);
            latestAudioLoudConf.set(0f);
            latestAudioLoudTsMs.set(0L);
            latestAudioLoudValid.set(false);
        };

        seekHandler.postDelayed(pendingSeekRunnable, 500);
    }

    /**
     * [WebVideo] 网页视频模式专用：建立 ExoPlayer，attach 到 PlayerView，监听状态/seek/错误。
     */
    private void setupExoPlayerForWebMode(Uri videoUri, Map<String, String> httpHeaders) {
        exoPlayerEngine = new ExoPlayerEngine(this, pcmBuffer, httpHeaders);
        ExoPlayer player = exoPlayerEngine.getPlayer();
        playerView.setPlayer(player);
        playerView.setUseController(true);

        player.addListener(new Player.Listener() {
            @Override
            public void onPlaybackStateChanged(int state) {
                if (state == Player.STATE_READY && !exoFirstReadyHandled) {
                    exoFirstReadyHandled = true;
                    onExoPlayerFirstReady();
                } else if (state == Player.STATE_ENDED) {
                    isVideoCompleted = true;
                    pauseAnalysis();
                    Log.d(TAG, "[网页视频] 视频播放完成，暂停同步分析");
                }
            }

            @Override
            public void onPositionDiscontinuity(Player.PositionInfo oldPosition,
                                                Player.PositionInfo newPosition,
                                                int reason) {
                if (reason == Player.DISCONTINUITY_REASON_SEEK
                        || reason == Player.DISCONTINUITY_REASON_SEEK_ADJUSTMENT) {
                    final long newMs = Math.max(0, newPosition.positionMs);
                    Log.i(TAG, "[网页视频] onPositionDiscontinuity seek -> " + newMs + "ms");
                    if (mainHandler != null) {
                        mainHandler.post(() -> handleWebVideoSeekComplete(newMs));
                    } else {
                        handleWebVideoSeekComplete(newMs);
                    }
                }
            }

            @Override
            public void onPlayerError(PlaybackException error) {
                Log.e(TAG, "[网页视频] ExoPlayer 错误 code=" + error.errorCode, error);

                // HTTP 层错误（典型：CDN 防盗链 403）把细节打出来，方便定位到底缺什么头
                String hint = "";
                Throwable cause = error.getCause();
                if (cause instanceof HttpDataSource.InvalidResponseCodeException) {
                    HttpDataSource.InvalidResponseCodeException http =
                            (HttpDataSource.InvalidResponseCodeException) cause;
                    Log.e(TAG, "[网页视频] HTTP " + http.responseCode
                            + " url=" + http.dataSpec.uri
                            + "\n  请求头(per-request)=" + http.dataSpec.httpRequestHeaders.keySet()
                            + "\n  响应头=" + http.headerFields);
                    hint = http.responseCode == 403
                            ? "（被 CDN 拒绝 403：防盗链校验未通过或视频已下架）"
                            : "（HTTP " + http.responseCode + "）";
                }

                Toast.makeText(VideoProcessActivity.this,
                        "网页视频播放失败: " + error.getErrorCodeName() + hint,
                        Toast.LENGTH_LONG).show();
            }
        });

        player.setPlayWhenReady(true);
        exoPlayerEngine.setUri(videoUri.toString());
        Log.d(TAG, "[网页视频] ExoPlayer 已 prepare: " + videoUri);
    }

    /**
     * [WebVideo] ExoPlayer 第一次进入 STATE_READY：相当于离线模式 onPrepared 回调。
     * 拿尺寸、按需起播 seek、启动多线程分析。
     */
    private void onExoPlayerFirstReady() {
        if (exoPlayerEngine == null) return;
        ExoPlayer player = exoPlayerEngine.getPlayer();
        if (player == null) return;

        VideoSize size = player.getVideoSize();
        originalWidth = size.width;
        originalHeight = size.height;
        long duration = player.getDuration();
        Log.d(TAG, "[网页视频] 首次 READY，尺寸: " + originalWidth + "x" + originalHeight
                + ", 时长: " + duration + "ms");

        // 哨兵：时长太短（< 60s）八成是 trickplay / 预览 / 广告 trailer，
        // 真视频几乎不会这么短。弹 Toast 警告，让用户返回重嗅。
        if (duration > 0 && duration < 60_000L) {
            Toast.makeText(this,
                    "视频时长仅 " + (duration / 1000) + " 秒，疑似预览/广告片段。\n"
                            + "请返回 WebView 等真视频开始播放后再点开始分析。",
                    Toast.LENGTH_LONG).show();
        }

        // PlayerView 内部用 resize_mode=fit 自动按比例缩放，这里 adjustVideoSize 仅对
        // 离线模式的 VideoView 起作用；为了不动它，给 originalWidth/Height 占位即可
        if (videoView.getVisibility() == View.VISIBLE) {
            adjustVideoSize(false);
        }

        if (webVideoStartPositionMs > 0) {
            Log.i(TAG, "[网页视频] 起播 seek 到 " + webVideoStartPositionMs
                    + "ms（来自 WebView currentTime）");
            player.seekTo(webVideoStartPositionMs);
        }

        startMultiThreadAnalysis();
    }

    /**
     * [WebVideo] seek 后清缓存：跟离线模式的 handleSeekComplete 大体一致，但因为我们
     * 不靠 videoFrameExtractor / audioDecoder，所以只需要重置共享状态。
     */
    private void handleWebVideoSeekComplete(long newPositionMs) {
        synchronized (videoLock) {
            frameWindow.clear();
            lastVideoInferenceTime = 0;
        }
        synchronized (audioLock) {
            pcmBuffer.reset();
        }
        synchronized (historyLock) {
            actionHistory.clear();
        }
        // [启动仲裁] 重置启动计数
        audioOnlyDoStreak = 0;
        conflictHoldTicks = 0;
        // [证据融合] seek 前的证据和结果快照都不能算进新位置
        resetEvidenceFusion();

        audioStartTime = newPositionMs;

        // 清空蓝牙控制动作缓存
        currentBluetoothState = "";
        pendingBluetoothState = "";
        latestBluetoothAction.set("");
        currentStateStartTime = 0;
        pendingStateStartTime = 0;
        currentLevel = 1;
        currentLevelSinceMs = 0;
        pendingLevel = null;
        pendingLevelSinceMs = 0;

        suspendedBluetoothState = "";
        suspendedLevel = 0;

        loudnessEstimator.reset();
        latestAudioLoudLevel.set(0);
        latestAudioLoudConf.set(0f);
        latestAudioLoudTsMs.set(0L);
        latestAudioLoudValid.set(false);

        // VAD 状态也重置，让起播位置后的"没声音"重新走一遍判断
        webVadActive = false;
        webVadConsecutiveSilentTicks = 0;
        webVadLastActiveMs = 0;

        Log.d(TAG, "[网页视频] seek 完成 -> " + newPositionMs + "ms，缓冲已清空");
    }

    /**
     * 统一的当前播放位置：网页模式走 ExoPlayer 的缓存值（任何线程安全），离线模式走 VideoView。
     */
    private long getCurrentPlaybackPosMs() {
        if (isWebVideoMode) {
            if (exoPlayerEngine == null) return 0L;
            return Math.max(0L, exoPlayerEngine.getCachedPositionMs());
        }
        if (videoView == null) return 0L;
        return videoView.getCurrentPosition();
    }

    /**
     * [WebVideo] 从 PlayerView 的 TextureView 直接读一帧。失败返回 null。
     */
    private Bitmap captureWebVideoFrameBitmap() {
        if (playerView == null) return null;
        View surfaceView = playerView.getVideoSurfaceView();
        if (!(surfaceView instanceof TextureView)) return null;
        TextureView tv = (TextureView) surfaceView;
        if (!tv.isAvailable()) return null;
        try {
            // 直接按模型输入尺寸读，省一次 createScaledBitmap
            return tv.getBitmap(VideoClassifierHelper.INPUT_SIZE,
                                VideoClassifierHelper.INPUT_SIZE);
        } catch (Exception e) {
            Log.w(TAG, "[网页视频] TextureView 抽帧失败", e);
            return null;
        }
    }

    /**
     * [WebVideo] 网页视频模式音频 VAD。
     * 在音频线程每秒触发一次。读取当前播放位置附近 0.25s 的 PCM，
     * 用 RMS/峰值 + 迟滞判定是否有声，决定 resume/pauseAnalysis。
     */
    private void runWebVideoVadTick() {
        long currentMs = getCurrentPlaybackPosMs();
        float[] samples = pcmBuffer.readWindowRelaxed(currentMs, 4000); // 0.25s @ 16kHz
        if (samples == null || samples.length == 0) {
            // PCM 未就绪：不切换状态，等下一拍再看
            return;
        }

        float rmsTh  = webVadActive ? WEB_VAD_KEEP_RMS_DB  : WEB_VAD_ACTIVATE_RMS_DB;
        float peakTh = webVadActive ? WEB_VAD_KEEP_PEAK_DB : WEB_VAD_ACTIVATE_PEAK_DB;

        double sumSq = 0;
        float peak = 0f;
        for (float s : samples) {
            sumSq += s * s;
            float a = Math.abs(s);
            if (a > peak) peak = a;
        }
        double rms = Math.sqrt(sumSq / samples.length);
        float rmsDb  = (float) (20.0 * Math.log10(Math.max(rms,  1e-9)));
        float peakDb = (float) (20.0 * Math.log10(Math.max(peak, 1e-9)));
        boolean active = rmsDb > rmsTh || peakDb > peakTh;

        long now = System.currentTimeMillis();
        if (now - lastWebVadLogMs > 5000) {
            lastWebVadLogMs = now;
            Log.d(TAG, String.format("[网页VAD] rms=%.1fdB peak=%.1fdB -> active=%b (state=%s)",
                    rmsDb, peakDb, active, webVadActive ? "ACTIVE" : "SILENT"));
        }

        if (active) {
            webVadLastActiveMs = now;
            webVadConsecutiveSilentTicks = 0;
            if (!webVadActive) {
                webVadActive = true;
                Log.i(TAG, "[网页VAD] 检测到音频活动，恢复分析");
                mainHandler.post(VideoProcessActivity.this::resumeAnalysis);
            }
        } else {
            webVadConsecutiveSilentTicks++;
            if (webVadActive
                    && webVadConsecutiveSilentTicks >= WEB_VAD_MIN_SILENT_TICKS
                    && (now - webVadLastActiveMs) > WEB_VAD_SILENCE_MS) {
                webVadActive = false;
                Log.i(TAG, "[网页VAD] 静音超过 " + WEB_VAD_SILENCE_MS + "ms，暂停分析");
                mainHandler.post(VideoProcessActivity.this::pauseAnalysis);
            }
        }
    }

    private void pauseAnalysis() {
        isAnalysisPaused.set(true);

        // ★ 新增：挂起当前蓝牙动作并发送停止信号
        if (!currentBluetoothState.isEmpty() && !"Noise".equals(currentBluetoothState)) {
            suspendedBluetoothState = currentBluetoothState;
            suspendedLevel = currentLevel;
            Log.i(TAG, String.format("[暂停] 挂起动作: %s, 档位: %d",
                    suspendedBluetoothState, suspendedLevel));
        }
        if (BLEManager.globalManager != null && BLEManager.globalManager.isConnected()
                && !BLEManager.globalManager.isPausedByLocal()) {
            BLEManager.globalManager.sendAction("Noise", 0);
            Log.i(TAG, "[暂停] 已发送停止信号(Noise)");
        }

        Log.i(TAG, "暂停分析");
    }

    private void resumeAnalysis() {
        isAnalysisPaused.set(false);

        // ★ 新增：恢复挂起的蓝牙动作
        if (!suspendedBluetoothState.isEmpty()) {
            Log.i(TAG, String.format("[恢复] 恢复动作: %s, 档位: %d",
                    suspendedBluetoothState, suspendedLevel));
            currentBluetoothState = suspendedBluetoothState;
            currentLevel = suspendedLevel;
            currentStateStartTime = System.currentTimeMillis();
            currentLevelSinceMs = System.currentTimeMillis();
            latestBluetoothAction.set(suspendedBluetoothState);

            if (BLEManager.globalManager != null && BLEManager.globalManager.isConnected()
                    && !BLEManager.globalManager.isPausedByLocal()) {
                BLEManager.globalManager.sendAction(suspendedBluetoothState, suspendedLevel);
                lastBluetoothSendTime = System.currentTimeMillis();
                lastSentLevel = suspendedLevel;
                Log.i(TAG, "[恢复] 已发送恢复动作: " + suspendedBluetoothState
                        + " 档位: " + suspendedLevel);
            }

            suspendedBluetoothState = "";
            suspendedLevel = 0;
        }

        Log.i(TAG, "恢复分析");
    }

    private void enterFullscreen() {
        isFullscreen = true;
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_FULLSCREEN |
                        View.SYSTEM_UI_FLAG_HIDE_NAVIGATION |
                        View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY |
                        View.SYSTEM_UI_FLAG_LAYOUT_STABLE |           // 🔴 新增
                        View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION |  // 🔴 新增
                        View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN        // 🔴 新增
        );

        // 🔴 新增：隐藏状态栏和导航栏
        if (getSupportActionBar() != null) {
            getSupportActionBar().hide();
        }

        btnFullscreen.setImageResource(R.drawable.ic_fullscreen_exit);

        // 🔴 新增：延迟调整视频尺寸，确保布局完成
        videoContainer.postDelayed(() -> adjustVideoSize(true), 100);
    }

    private void exitFullscreen() {
        isFullscreen = false;
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_VISIBLE);

        // 显示状态栏
        if (getSupportActionBar() != null) {
            getSupportActionBar().show();
        }

        btnFullscreen.setImageResource(R.drawable.ic_fullscreen);

        // 延迟调整视频尺寸，确保布局完成
        videoContainer.postDelayed(() -> adjustVideoSize(false), 100);
    }

    // 添加调整视频尺寸的方法
    private void adjustVideoSize(boolean isFullscreen) {
        if (originalWidth == 0 || originalHeight == 0) return;

        // 获取容器尺寸
        int containerWidth = videoContainer.getWidth();
        int containerHeight = videoContainer.getHeight();

        if (containerWidth == 0 || containerHeight == 0) {
            // 如果容器尺寸还未确定，延迟执行
            videoContainer.post(() -> adjustVideoSize(isFullscreen));
            return;
        }

        float videoAspectRatio = (float) originalWidth / originalHeight;
        float containerAspectRatio = (float) containerWidth / containerHeight;

        int finalWidth;
        int finalHeight;

        if (isFullscreen) {
            // 全屏模式：尽可能填充屏幕
            if (videoAspectRatio > containerAspectRatio) {
                // 视频更宽，以宽度为准
                finalWidth = containerWidth;
                finalHeight = (int) (containerWidth / videoAspectRatio);
            } else {
                // 视频更高，以高度为准
                finalHeight = containerHeight;
                finalWidth = (int) (containerHeight * videoAspectRatio);
            }
        } else {
            // 竖屏模式：保持原有逻辑
            if (videoAspectRatio > containerAspectRatio) {
                finalWidth = containerWidth;
                finalHeight = (int) (containerWidth / videoAspectRatio);
            } else {
                finalHeight = containerHeight;
                finalWidth = (int) (containerHeight * videoAspectRatio);
            }
        }

        // 更新 VideoView 的布局参数
        FrameLayout.LayoutParams params = (FrameLayout.LayoutParams) videoView.getLayoutParams();
        params.width = finalWidth;
        params.height = finalHeight;
        params.gravity = Gravity.CENTER;
        videoView.setLayoutParams(params);

        Log.d(TAG, String.format("调整视频尺寸 - 容器: %dx%d, 视频: %dx%d, 最终: %dx%d",
                containerWidth, containerHeight, originalWidth, originalHeight, finalWidth, finalHeight));
    }

    public void onConfigurationChanged(Configuration newConfig) {
        super.onConfigurationChanged(newConfig);

        // 屏幕方向改变时重新调整视频尺寸
        videoContainer.post(() -> adjustVideoSize(isFullscreen));
    }


    @Override
    protected void onDestroy() {
        Log.d(TAG, "onDestroy: 准备释放资源...");

        // [评测] 收尾并落盘
        BenchLogger.get().stop();

        // 设置停止标志
        shouldStop.set(true);

        // ★ 新增：退出时发送停止信号
        if (BLEManager.globalManager != null && BLEManager.globalManager.isConnected()) {
            BLEManager.globalManager.sendAction("Noise", 0);
            Log.i(TAG, "[退出] 已发送停止信号(Noise)");
        }

        // 停止所有线程
        if (videoThread != null) {
            videoThread.quitSafely();
            try {
                videoThread.join(1000);
            } catch (InterruptedException e) {
                Log.e(TAG, "等待视频线程结束时被中断", e);
            }
        }

        if (audioThread != null) {
            audioThread.quitSafely();
            try {
                audioThread.join(1000);
            } catch (InterruptedException e) {
                Log.e(TAG, "等待音频线程结束时被中断", e);
            }
        }

        // 清理Handler回调
        if (mainHandler != null) {
            mainHandler.removeCallbacksAndMessages(null);
        }

        if (playStateHandler != null && playStateChecker != null) {
            playStateHandler.removeCallbacks(playStateChecker);
        }

        // 释放资源
        if (videoFrameExtractor != null) {
            videoFrameExtractor.release();
        }

        if (videoClassifier != null) {
            videoClassifier.close();
        }

        if (audioDecoder != null) {
            audioDecoder.stop();
        }

        if (audioHelper != null) {
            audioHelper.close();
        }

        // [WebVideo] 释放 ExoPlayer
        if (playerView != null) {
            try { playerView.setPlayer(null); } catch (Exception ignored) {}
        }
        if (exoPlayerEngine != null) {
            exoPlayerEngine.release();
            exoPlayerEngine = null;
        }

        super.onDestroy();
    }
}