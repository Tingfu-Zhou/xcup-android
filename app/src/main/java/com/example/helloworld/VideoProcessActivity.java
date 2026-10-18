package com.example.helloworld;

import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Log;
import android.widget.MediaController;
import android.widget.TextView;
import android.widget.VideoView;
import android.graphics.Bitmap;

import androidx.appcompat.app.AppCompatActivity;

import android.widget.Toast;

import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicBoolean;

import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.pm.ApplicationInfo;
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

    // 显示蓝牙发送的动作
    private final AtomicReference<String> latestBluetoothAction = new AtomicReference<>("");

    // 模型结果快照：判定（动作串 + 置信度）、原始概率、产出时刻作为<b>一个</b>对象原子发布，融合线程只读这一个对象。
    // 融合仲裁按产出时刻去重，动作和时间戳必须来自同一次推理；评测 CSV 的概率和 v_age/a_age 也取自它。
    private final AtomicReference<ResultSnap> latestVideoSnap = new AtomicReference<>(null);
    private final AtomicReference<ResultSnap> latestAudioSnap = new AtomicReference<>(null);

    /**
     * 一次模型推理的结果快照。action 是分组概率判定（"do"/"Noise"/"" unclear），probs 为 null 表示推理失败。
     * winEndPosMs：模型窗口结尾的播放位置。音频是读 PCM 时的播放位置（窗口 = [winEndPosMs-2s, winEndPosMs]），写进评测 CSV 的
     * a_pos 列；视频是窗口里最后一帧的时刻，写进 v_pos 列。bench/tools/audio_rescore.py / video_rescore.py 离线重算时靠它们还原模型输入。
     * framePosMs 只有视频有：窗口里 12 帧各自的时刻（离线模式是帧的 pts，网页模式是抓帧时的播放位置），写进 v_fpts 列。
     * cue / cueFeat 只有音频有：YAMNet AudioSet 分数判的线索（{@link AudioCue}）和分组分数，交给融合仲裁、写进评测 CSV。
     */
    private static final class ResultSnap {
        final String action;
        final float conf;
        final float[] probs;
        final long tsMs;            // 结果产出时刻（墙钟）
        final long winEndPosMs;     // 模型窗口结尾的播放位置（见上）
        final int cue;              // AudioCue.*，视频 / 拿不到分数时 UNKNOWN
        final float[] cueFeat;      // AudioCue.features，拿不到时 null
        final long[] framePosMs;    // 视频窗口各帧的时刻（从旧到新），音频为 null

        ResultSnap(String action, float conf, float[] probs, long tsMs, long winEndPosMs,
                   int cue, float[] cueFeat) {
            this(action, conf, probs, tsMs, winEndPosMs, cue, cueFeat, null);
        }

        ResultSnap(String action, float conf, float[] probs, long tsMs, long winEndPosMs,
                   int cue, float[] cueFeat, long[] framePosMs) {
            this.action = action;
            this.conf = conf;
            this.probs = probs;
            this.tsMs = tsMs;
            this.winEndPosMs = winEndPosMs;
            this.cue = cue;
            this.cueFeat = cueFeat;
            this.framePosMs = framePosMs;
        }
    }

    // [融合仲裁] 启动 / 维持 / 停止 分开判断（规则见 FusionArbiter）。2026-10-07 起替代原来的平滑投票 smoothedFusion
    // 和启动仲裁 applyDoStartGate（都已删除）。只在融合线程（主线程）访问；seek / 恢复分析时 reset。
    private final FusionArbiter fusionArbiter = new FusionArbiter();

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

    // [定时加档] 分析累计播放满用户选定的时间后，发给设备的档位再加 1 / 2 档（设置在手动电子菜单，规则见 LevelBoost）。
    // currentLevel 仍是分析档位，只在发送时加（boostLevel）；lastSentLevel 记的是加档后实际发出去的档位。只在主线程访问。
    private LevelBoost levelBoost;
    private int levelBoostNow = 0;                  // 此刻加几档（没打开 / 没到点为 0）：融合 tick 里更新，视频播完清零

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

    // 帧缓存环形缓冲：最近 3 秒、间隔 250ms 采样的 12 帧（已是模型输入 160×160），frameTimeWindow 是每帧的时刻
    // （离线模式：帧的 pts；网页模式：抓帧时的播放位置），与 frameWindow 一一对应
    private final ArrayDeque<Bitmap> frameWindow = new ArrayDeque<>();
    private final ArrayDeque<Long> frameTimeWindow = new ArrayDeque<>();
    private static final int FRAME_WINDOW_SIZE = VideoClassifierHelper.NUM_FRAMES;

    // [抽帧] 离线模式的抽帧线程：VideoFrameExtractor（MediaCodec + EGL）只在这个线程里用（EGL 上下文绑定在线程上）。
    // 2026-10-14 之前抽帧在主线程做（每 250ms 占约 43ms），而且拿到的只是关键帧（见 VideoFrameExtractor 类注释）。
    private HandlerThread frameThread;
    private Handler frameHandler;
    // 上一帧还没取完（seek 后追赶时可能超过 250ms）就不再排新的请求，免得请求堆积、取到的都是过时的位置
    private final AtomicBoolean frameRequestInFlight = new AtomicBoolean(false);
    // seek 时加一：seek 之前请求、之后才送到视频线程的帧一律丢掉
    private final AtomicInteger frameGeneration = new AtomicInteger(0);
    // [网页视频] TextureView 截帧的暂存图（短边 448 = 2×224）和像素缓冲，只在主线程用
    private static final int WEB_STAGE_SHORT_SIDE = 448;
    private Bitmap webStageBitmap;
    private int[] webStagePixels;

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
    // 网页视频模式没有 MediaController 和播放状态轮询：分析的开/停由音频 VAD（参考在线模式）决定，
    // 播放器暂停（用户暂停、来电等）另外处理，见 onWebPlayerPlayIntentChanged
    private boolean isWebVideoMode = false;
    // VAD 状态（音频线程写，主线程也读写）
    private volatile boolean webVadActive = false;
    // [网页视频] 播放器是否处于暂停（用户暂停，或来电等抢走音频焦点）。只在主线程访问
    private boolean webPlayerPaused = false;
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

        // 调试悬浮窗只在 debug 包（Android Studio 直接 Run 装的包）显示，release 包隐藏。
        // 与 BenchLogger 一样看 FLAG_DEBUGGABLE，不依赖 BuildConfig。
        boolean debuggable = (getApplicationInfo().flags & ApplicationInfo.FLAG_DEBUGGABLE) != 0;
        findViewById(R.id.debugPanel).setVisibility(debuggable ? View.VISIBLE : View.GONE);

        // [定时加档] 进入页面时读一次手动电子菜单里的设置
        levelBoost = ManualControlActivity.loadLevelBoost(this);

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
                // 构造函数只打开 MediaExtractor；解码器和 EGL 在抽帧线程第一次取帧时创建，之后的调用也都在那个线程
                videoFrameExtractor = new VideoFrameExtractor(this, videoUri, null);
                Log.d(TAG, "onCreate: VideoFrameExtractor 初始化成功");
            } catch (IOException e) {
                Log.e(TAG, "onCreate: VideoFrameExtractor 初始化失败 uri=" + videoUri, e);
                Toast.makeText(this, "视频解析失败：" + e.getMessage(), Toast.LENGTH_LONG).show();
                finish();
                return;
            }
            frameThread = new HandlerThread("FrameExtractThread");
            frameThread.start();
            frameHandler = new Handler(frameThread.getLooper());

            videoView.setMediaController(new MediaController(this));
            videoView.requestFocus();

            videoView.setOnCompletionListener(mp -> {
                isVideoCompleted = true;
                pauseAnalysis();
                resetLevelBoost(); // [定时加档] 视频播完，从头计时
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
                            resumeAnalysis(false);
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
     * 每 250ms 抽取并缓存一帧；缓存窗口满 12 帧（3 秒）后，每 500ms 执行一次推理。
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
                    final int gen = frameGeneration.get();
                    if (isWebVideoMode) {
                        // 网页模式：TextureView 只能在主线程读
                        mainHandler.post(() -> {
                            try {
                                long tExtractStart = System.currentTimeMillis();
                                long posMs = getCurrentPlaybackPosMs();
                                Bitmap frame = captureWebVideoFrameBitmap();
                                Log.d(TAG, "[计时] 📸 抽帧耗时: "
                                        + (System.currentTimeMillis() - tExtractStart) + " ms");
                                if (frame != null) {
                                    videoHandler.post(() -> cacheFrameAndInfer(frame, posMs, gen));
                                } else {
                                    Log.w(TAG, "[主线程] [同步分析] 未能抽取到视频帧");
                                }
                            } catch (Exception e) {
                                Log.e(TAG, "[主线程] 抽帧异常：", e);
                            }
                        });
                    } else if (frameHandler != null && frameRequestInFlight.compareAndSet(false, true)) {
                        // 离线模式：抽帧线程取“此刻播放位置的那一帧”（位置在抽帧线程里读，排队等待不会让它过时）
                        frameHandler.post(() -> {
                            try {
                                if (shouldStop.get()) return;
                                long tExtractStart = System.currentTimeMillis();
                                long posMs = getCurrentPlaybackPosMs();
                                VideoFrameExtractor.Frame fr = videoFrameExtractor.getFrameAt(posMs * 1000L);
                                Log.d(TAG, "[计时] 📸 抽帧耗时: " + (System.currentTimeMillis() - tExtractStart)
                                        + " ms（位置 " + posMs + "ms"
                                        + (fr == null ? "，未取到）" : "，取到 " + fr.ptsUs / 1000 + "ms 的帧）"));
                                if (fr != null) {
                                    videoHandler.post(() -> cacheFrameAndInfer(fr.bitmap, fr.ptsUs / 1000L, gen));
                                }
                            } catch (Exception e) {
                                Log.e(TAG, "[抽帧线程] 抽帧异常：", e);
                            } finally {
                                frameRequestInFlight.set(false);
                            }
                        });
                    }
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
     * 缓存一帧到滑动窗口；窗口满 12 帧且距上次推理 ≥ 500ms 时执行一次推理。
     *
     * @param frame      模型输入（160×160，已按 FramePreprocessor 裁剪缩放）
     * @param framePosMs 这一帧的时刻（离线：pts；网页：抓帧时的播放位置）
     * @param gen        请求这一帧时的 frameGeneration；之后发生过 seek 就丢掉
     */
    private void cacheFrameAndInfer(Bitmap frame, long framePosMs, int gen) {
        synchronized (videoLock) {
            if (gen != frameGeneration.get()) {
                Log.d(TAG, "[视频线程] seek 之前请求的帧（" + framePosMs + "ms），丢弃");
                return;
            }
            // 同一帧不重复放（播放卡住时位置不动）：窗口应当是 12 个不同时刻
            Long lastPos = frameTimeWindow.peekLast();
            if (lastPos != null && lastPos == framePosMs) {
                return;
            }
            Bitmap scaled = (frame.getWidth() == VideoClassifierHelper.INPUT_SIZE
                    && frame.getHeight() == VideoClassifierHelper.INPUT_SIZE)
                    ? frame : VideoClassifierHelper.toModelBitmap(frame, null);
            frameWindow.add(scaled);
            frameTimeWindow.add(framePosMs);
            while (frameWindow.size() > FRAME_WINDOW_SIZE) {
                frameWindow.poll();
                frameTimeWindow.poll();
            }

            // 滑动窗口未满 3 秒，暂不推理
            if (frameWindow.size() < FRAME_WINDOW_SIZE) {
                Log.d(TAG, "[视频线程] 帧窗口预热中 (" + frameWindow.size()
                        + "/" + FRAME_WINDOW_SIZE + ")");
                return;
            }

            // 推理步长 500ms
            long now = System.currentTimeMillis();
            if (now - lastVideoInferenceTime < VIDEO_INFERENCE_INTERVAL) {
                return;
            }
            lastVideoInferenceTime = now;

            Bitmap[] frames = frameWindow.toArray(new Bitmap[0]);
            long[] framePos = new long[frameTimeWindow.size()];
            int k = 0;
            for (long t : frameTimeWindow) framePos[k++] = t;
            long tInferStart = System.currentTimeMillis();
            VideoClassifierHelper.Result result = videoClassifier.predict(frames);
            Log.d(TAG, "[计时] 🧠 MobileNetV3Small 推理耗时: "
                    + (System.currentTimeMillis() - tInferStart) + " ms");

            applyVideoResult(result, framePos);
        }
    }

    /**
     * 将三分类结果映射为视频动作并写入共享结果（分组概率决策）。
     * P(do)=P(oral)+P(sex) >= 阈值 -> "do"（转）；P(plot) >= 阈值 -> "Noise"（不转）；
     * 其余 -> unclear -> ""（置信度 0）。置信度取分组后的概率。
     *
     * @param framePos 窗口里各帧的时刻（从旧到新），写进评测 CSV 的 v_pos / v_fpts，离线重算靠它还原模型输入
     */
    private void applyVideoResult(VideoClassifierHelper.Result result, long[] framePos) {
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

        // 判定 + 原始概率 + 产出时刻打包成一个快照原子发布（推理失败时 probs 为 null）
        long resultTs = System.currentTimeMillis();
        boolean probsOk = result != null && result.index >= 0
                && result.probs != null && result.probs.length >= 3;
        latestVideoSnap.set(new ResultSnap(actionClass, confidence,
                probsOk ? result.probs.clone() : null, resultTs,
                framePos.length > 0 ? framePos[framePos.length - 1] : -1, AudioCue.UNKNOWN, null, framePos));

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

                                // [音频线索] YAMNet 自带的 AudioSet 分数判"这一刻音频靠不靠得住"（对白 / 听不出 / 人声 …），
                                // 只给融合仲裁的启动判断用，不改上面的分类判定（规则见 AudioCue、FusionArbiter）
                                float[] cueFeat = AudioCue.features(result.audioSet);
                                int cue = AudioCue.classify(cueFeat);

                                // 判定 + 原始概率 + 产出时刻 + 读窗位置 + 线索打包成一个快照原子发布
                                // （评测 CSV 的 a_p_* / a_age / a_pos / a_cue* 取自它）
                                long resultTs = System.currentTimeMillis();
                                boolean probsOk = result.index >= 0 && result.probs.length >= 3;
                                latestAudioSnap.set(new ResultSnap(action, confidence,
                                        probsOk ? result.probs.clone() : null, resultTs, currentMs, cue, cueFeat));

                                // UI更新
                                final String displayText = (action.isEmpty()
                                        ? "A: unclear"
                                        : String.format("A: %s (p=%.2f)", action, confidence))
                                        + (cue == AudioCue.UNKNOWN ? "" : " · " + AudioCue.label(cue));
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
                    levelBoost.pause(SystemClock.elapsedRealtime()); // [定时加档] 暂停期间不计时
                    mainHandler.postDelayed(this, MAIN_FUSION_INTERVAL);
                    return;
                }
                long t0 = System.currentTimeMillis();

                // 读取最新的分析结果：每个模态只读一个快照对象，动作、置信度、产出时刻保证来自同一次推理
                ResultSnap vSnap = latestVideoSnap.get();
                ResultSnap aSnap = latestAudioSnap.get();
                String videoAction = vSnap == null ? "" : vSnap.action;
                String audioAction = aSnap == null ? "" : aSnap.action;
                float videoConf = vSnap == null ? 0f : vSnap.conf;
                float audioConf = aSnap == null ? 0f : aSnap.conf;
                long videoTime = vSnap == null ? 0L : vSnap.tsMs;
                long audioTime = aSnap == null ? 0L : aSnap.tsMs;

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

                // [融合仲裁] 启动 / 维持 / 停止 分开判断，证据按"新推理结果"计数（按快照的产出时刻去重；
                // 与评测 CSV 的 v_age/a_age 同源，离线回放才能逐 tick 复现）。音频线索与音频结果同一个快照
                int audioCue = aSnap == null ? AudioCue.UNKNOWN : aSnap.cue;
                String finalAction = fusionArbiter.decide(currentTime,
                        videoAction, videoConf, videoTime > 0 ? videoTime : -1,
                        audioAction, audioConf, audioTime > 0 ? audioTime : -1,
                        audioCue, isSexAction(currentBluetoothState));
                // 临时采用音频节律作为最终节律
                // 将“最终节律计算”封装为独立方法，便于后续替换为音视频融合节律
                int finalFreq = computeFinalFreq(audioFreq, audioFreqConf);
                // [定时加档] 接着计时，算出此刻该加几档（updateBluetoothState 发送时加在分析档位上）
                updateLevelBoost();

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
                            aSnap == null ? -1 : aSnap.winEndPosMs,
                            Float.isNaN(audioFreq) ? -1 : Math.round(audioFreq),
                            audioFreqConf, audioFreqValid, audioFreqAge,
                            finalAction, finalFreq,
                            currentBluetoothState, currentLevel,
                            fusionArbiter.lastReason(),
                            audioCue, aSnap == null ? null : aSnap.cueFeat,
                            vSnap == null ? -1 : vSnap.winEndPosMs, vSnap == null ? null : vSnap.framePosMs);
                }

                // 更新UI, 显示蓝牙实际发送的动作
                String bluetoothAction = latestBluetoothAction.get();
                if (!bluetoothAction.isEmpty()) {
                    tvOverlay.setText("蓝牙: " + bluetoothAction + "节奏：" + currentLevel
                            + (levelBoostNow > 0 ? "（加档 +" + levelBoostNow + "）" : ""));
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
            // [定时加档] 实际发给设备的档位：到点后在分析档位上加档（0 档不加）；没打开 / 没到点时与 levelToSend 相同
            int boostedLevel = boostLevel(levelToSend);

            Log.i(TAG, String.format(
                    "[蓝牙] [同步分析] ✅ 发送指令: action=%s, level=%d, 已稳定=%dms",
                    pendingBluetoothState,
                    boostedLevel,
                    currentTime - pendingStateStartTime
            ));

            latestBluetoothAction.set(pendingBluetoothState);
            BenchLogger.get().noteSend(pendingBluetoothState, boostedLevel);

            if (BLEManager.globalManager != null && BLEManager.globalManager.isConnected()) {
                BLEManager.globalManager.sendAction(pendingBluetoothState, boostedLevel);
                Log.i(TAG, "[蓝牙] [同步分析] ✅ 已通过BLE发送指令(动作切换/含档位)");
                Log.d(TAG, "[音频节律] ✅ 发送节律：" + boostedLevel);
            } else {
                Log.w(TAG, "[蓝牙] BLE未连接，仅更新UI显示");
            }

            currentBluetoothState = pendingBluetoothState;
            currentStateStartTime = currentTime;
            lastBluetoothSendTime = currentTime;

            // 关键同步：
            // 保证 currentLevel / lastSentLevel / 实际发送档位一致。
            // [定时加档] currentLevel 记分析档位，lastSentLevel 记实际发出去的（加档后的）档位
            if (isSexAction(currentBluetoothState)) {
                currentLevel = levelToSend;
                currentLevelSinceMs = currentTime;
                lastSentLevel = boostedLevel;
            } else {
                lastSentLevel = 0;
            }

            return;
        }


        // =====================[ 情况 B：动作未变，但档位变化 ]=====================
        // 动作不变，例如 do -> do，但档位从 8 降到 4。
        // 这种情况需要重发同一个动作，携带新档位。
        // [定时加档] 比较的是发给设备的档位：到点加档时分析档位没变，也会在这里按发送间隔补发一次
        int boostedLevel = boostLevel(currentLevel);
        boolean levelChanged = supportsLevel && (boostedLevel != lastSentLevel);
        boolean gapOk = (currentTime - lastBluetoothSendTime) >= BLUETOOTH_SEND_INTERVAL;

        Log.d(TAG, String.format(
                "[档位发送判断] action=%s, supportsLevel=%s, currentLevel=%d, boost=%d, lastSentLevel=%d, levelChanged=%s, gapOk=%s, gap=%dms",
                currentBluetoothState,
                supportsLevel,
                currentLevel,
                levelBoostNow,
                lastSentLevel,
                levelChanged,
                gapOk,
                currentTime - lastBluetoothSendTime
        ));

        if (levelChanged && gapOk) {
            int levelToSend = boostedLevel;

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
     * [评测] 把当前这组可调参数拼成一行，写进 run CSV 的文件头。
     *
     * <p>没有这一行，事后就分不清某份 dump 是哪组参数跑出来的，“每改一个参数都能量化
     * 对比”也就无从谈起。加了新参数记得同步加到这里。
     */
    private String benchParamsLine() {
        return String.format(Locale.US,
                "VIDEO_DO=%.2f VIDEO_PLOT=%.2f AUDIO_DO=%.2f AUDIO_NOISE=%.2f "
                        + "BLE_MIN_DUR=%d BLE_GAP=%d LEVEL_STABLE=%d LEVEL_MIN_DUR=%d "
                        + "FUSION_TICK=%d VIDEO_INFER=%d AUDIO_INFER=%d AUDIO_PIPE=%s VIDEO_PIPE=%s ",
                VIDEO_DO_PROB_THRESHOLD, VIDEO_PLOT_PROB_THRESHOLD,
                AUDIO_DO_PROB_THRESHOLD, AUDIO_NOISE_PROB_THRESHOLD,
                BLUETOOTH_MIN_DURATION, BLUETOOTH_SEND_INTERVAL,
                LEVEL_STABLE_MS, LEVEL_MIN_DUR_MS,
                MAIN_FUSION_INTERVAL, VIDEO_INFERENCE_INTERVAL, AUDIO_INFERENCE_INTERVAL,
                isWebVideoMode ? "web" : "fixed",
                // 视频输入：抽帧方式_几何（exact_crop = 修复后的离线模式；没有这一项的旧 run 是 keyframe + 整帧压扁）
                (isWebVideoMode ? "web_" : "exact_") + FramePreprocessor.geometryName())
                + FusionArbiter.paramsLine() + " " + AudioCue.paramsLine()
                // 定时加档：打开时到点后 sent_level 是加档后的档位（final_freq / ble_level 仍是分析档位）
                + " " + levelBoost.paramsLine();
    }



    /**
     * 清空融合仲裁的证据（seek / 恢复播放），并在评测 CSV 里留下 reset 记号，离线回放才能同样处理。
     *
     * @param seek     这次是 seek（调用方还会清空蓝牙状态和投票窗，回放时一并清）
     * @param warmupMs 再丢掉 reset 后这么久内产出的结果（静音暂停后恢复时用）
     */
    private void resetFusionArbiter(boolean seek, long warmupMs) {
        long now = System.currentTimeMillis();
        fusionArbiter.reset(now, warmupMs);
        BenchLogger.get().noteArbReset(now, warmupMs, seek);
    }

    /** [定时加档] 融合循环在跑：接着计时，更新此刻该加几档。计时用单调时钟，改系统时间不影响 */
    private void updateLevelBoost() {
        long now = SystemClock.elapsedRealtime();
        levelBoost.resume(now);
        levelBoostNow = levelBoost.boostAt(now);
    }

    /** [定时加档] 视频播完：从头计时，之后发出去的档位不再加 */
    private void resetLevelBoost() {
        levelBoost.reset();
        levelBoostNow = 0;
    }

    /** [定时加档] 分析档位 → 发给设备的档位：到点后加 levelBoostNow 档（0 档不加，最高 BLEManager.LEVEL_MAX） */
    private int boostLevel(int level) {
        return LevelBoost.apply(level, levelBoostNow, BLEManager.LEVEL_MAX);
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

            // 清空帧缓存和 pcm 缓冲，避免读取到“前一个时间点”的旧数据；seek 之前请求、还在路上的帧也作废
            synchronized (videoLock) {
                frameWindow.clear();
                frameTimeWindow.clear();
                frameGeneration.incrementAndGet();
                lastVideoInferenceTime = 0;
            }

            synchronized (audioLock) {
                pcmBuffer.reset();
            }

            // [融合仲裁] seek 后清空证据：seek 之前产出的结果一律不算
            resetFusionArbiter(true, 0);

            // 抽帧器下一次取帧时按当时的播放位置重新定位（从前一个关键帧解到目标帧）。抽帧器只能在抽帧线程里用
            if (frameHandler != null) {
                frameHandler.post(videoFrameExtractor::requestSeek);
            }
            // decoder 负责重新 seek 解码
            audioDecoder.seekTo(currentMs);
            Log.d(TAG, "🎯 audioDecoder.seekTo() 调用，当前视频播放时间: " + currentMs);
            audioStartTime = currentMs;

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

            // 如果之前已经播到结尾，清零标记并恢复分析。放在清空蓝牙状态 / 挂起动作之后：
            // 否则 resumeAnalysis 会先把播完前挂起的旧动作发出去，紧接着状态又被清成 ""，app 和设备就对不上了。
            if (isVideoCompleted) {
                isVideoCompleted = false;
                // 只有当前已经在播放，才立刻恢复分析；否则继续保持暂停
                if (videoView.isPlaying()) {
                    resumeAnalysis(false);
                }
            }

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
                    resetLevelBoost(); // [定时加档] 视频播完，从头计时
                    Log.d(TAG, "[网页视频] 视频播放完成，暂停同步分析");
                }
            }

            @Override
            public void onPlayWhenReadyChanged(boolean playWhenReady, int reason) {
                onWebPlayerPlayIntentChanged(player);
            }

            @Override
            public void onPlaybackSuppressionReasonChanged(int playbackSuppressionReason) {
                onWebPlayerPlayIntentChanged(player);
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
     * [网页视频] 播放器"要不要播"变了：用户暂停 / 继续，或来电等抢走音频焦点（ExoPlayer 暂缓播放，playWhenReady 不变）。
     *
     * <p>暂停后 PCM 不再写入，VAD 和音频模型读到的一直是暂停前最后那几秒，靠 VAD 判断不出暂停：原来分析会接着跑，
     * 暂停在动作上时马达也接着转。这里和离线模式一样处理：暂停时 pauseAnalysis（有证据的动作挂起、发 Noise）；
     * 继续播放时 VAD 若还认为有声（暂停前在响）就直接 resumeAnalysis 接着转，否则照旧等 VAD 听到声音再恢复。
     * 网络卡顿缓冲不算暂停（ExoPlayer 自己会接着播）。onPlayWhenReadyChanged 会重复报同一个值，这里只看状态变化。</p>
     */
    private void onWebPlayerPlayIntentChanged(Player player) {
        boolean paused = !player.getPlayWhenReady()
                || player.getPlaybackSuppressionReason() != Player.PLAYBACK_SUPPRESSION_REASON_NONE;
        if (paused == webPlayerPaused) return;
        webPlayerPaused = paused;
        if (paused) {
            Log.i(TAG, "[网页视频] 播放器暂停，暂停分析");
            pauseAnalysis();
        } else if (webVadActive) {
            Log.i(TAG, "[网页视频] 播放器继续播放，恢复分析");
            resumeAnalysis(false);
        }
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
            frameTimeWindow.clear();
            frameGeneration.incrementAndGet();
            lastVideoInferenceTime = 0;
        }
        synchronized (audioLock) {
            pcmBuffer.reset();
        }
        // [融合仲裁] seek 后清空证据：seek 之前产出的结果一律不算
        resetFusionArbiter(true, 0);

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
     * [WebVideo] 从 PlayerView 的 TextureView 读一帧，处理成模型输入（160×160）。失败返回 null。只能在主线程调用。
     *
     * <p>2026-10-14 之前直接 getBitmap(160, 160)：整帧压扁成正方形，与训练（居中裁正方形）不一致。现在按 TextureView 的
     * 宽高比截一张短边 448（= 2×224）的图，再交给 FramePreprocessor 居中裁剪、面积缩小。TextureView 在 PlayerView 里按视频
     * 宽高比摆放（resize_mode=fit），所以截到的就是完整的视频画面。</p>
     */
    private Bitmap captureWebVideoFrameBitmap() {
        if (playerView == null) return null;
        View surfaceView = playerView.getVideoSurfaceView();
        if (!(surfaceView instanceof TextureView)) return null;
        TextureView tv = (TextureView) surfaceView;
        if (!tv.isAvailable() || tv.getWidth() <= 0 || tv.getHeight() <= 0) return null;
        try {
            int vw = tv.getWidth(), vh = tv.getHeight();
            int sw, sh;
            if (vw >= vh) {
                sh = WEB_STAGE_SHORT_SIDE;
                sw = Math.round(WEB_STAGE_SHORT_SIDE * (float) vw / vh);
            } else {
                sw = WEB_STAGE_SHORT_SIDE;
                sh = Math.round(WEB_STAGE_SHORT_SIDE * (float) vh / vw);
            }
            if (webStageBitmap == null || webStageBitmap.getWidth() != sw || webStageBitmap.getHeight() != sh) {
                webStageBitmap = Bitmap.createBitmap(sw, sh, Bitmap.Config.ARGB_8888);
                webStagePixels = new int[sw * sh];
            }
            tv.getBitmap(webStageBitmap);
            return VideoClassifierHelper.toModelBitmap(webStageBitmap, webStagePixels);
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
                mainHandler.post(() -> {
                    // 播放器暂停期间不由 VAD 恢复（PCM 不再更新，VAD 听到的是暂停前的声音），继续播放时再处理
                    if (!webPlayerPaused) resumeAnalysis(true);
                });
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
            if (fusionArbiter.lastDoHadEvidence()) {
                suspendedBluetoothState = currentBluetoothState;
                suspendedLevel = currentLevel;
                Log.i(TAG, String.format("[暂停] 挂起动作: %s, 档位: %d",
                        suspendedBluetoothState, suspendedLevel));
            } else {
                // [融合仲裁] 马达此刻只是仲裁器在"等结果"（刚 seek / 恢复后的冷启动，或看不清听不清），没有证据撑着：
                // 不带过暂停，状态同步成 Noise（下面照常发 Noise），恢复后从头判断。否则网页视频模式下
                // "静音暂停 → 一声短响恢复 → 还在等结果时又静音暂停"会让马达跟着每一声短响一直转。
                long now = System.currentTimeMillis();
                currentBluetoothState = "Noise";
                currentStateStartTime = now;
                pendingBluetoothState = "Noise";
                pendingStateStartTime = now;
                latestBluetoothAction.set("Noise");
                Log.i(TAG, "[暂停] 马达没有证据撑着（" + fusionArbiter.lastReason() + "），不挂起，恢复后重新判断");
            }
        }
        if (BLEManager.globalManager != null && BLEManager.globalManager.isConnected()
                && !BLEManager.globalManager.isPausedByLocal()) {
            BLEManager.globalManager.sendAction("Noise", 0);
            Log.i(TAG, "[暂停] 已发送停止信号(Noise)");
        }

        Log.i(TAG, "暂停分析");
    }

    /**
     * @param afterSilence 网页视频模式 VAD 因静音暂停后、声音回来时恢复：模型窗口里还混着暂停前的静音，
     *                     融合仲裁先丢掉恢复后 2 秒内产出的结果
     */
    private void resumeAnalysis(boolean afterSilence) {
        isAnalysisPaused.set(false);
        // [融合仲裁] 暂停前的证据已经过时，清空重来。恢复后马达若仍在转，仲裁会先等两路攒够新结果
        // （wait_sparse，最多 6 秒），不会因为某一路刚出的一个"不转"就停。
        resetFusionArbiter(false, afterSilence ? FusionArbiter.SILENCE_RESUME_WARMUP_MS : 0);

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
                // [定时加档] 暂停不清零计时：已经到点的，恢复后接着按加档后的档位转
                int boostedLevel = boostLevel(suspendedLevel);
                BLEManager.globalManager.sendAction(suspendedBluetoothState, boostedLevel);
                lastBluetoothSendTime = System.currentTimeMillis();
                lastSentLevel = boostedLevel;
                Log.i(TAG, "[恢复] 已发送恢复动作: " + suspendedBluetoothState
                        + " 档位: " + boostedLevel);
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

        // 抽帧器（MediaCodec + EGL）必须在抽帧线程里释放；quitSafely 会先跑完已经排进去的消息
        if (frameThread != null) {
            final VideoFrameExtractor extractor = videoFrameExtractor;
            if (extractor != null) {
                frameHandler.post(extractor::release);
            }
            frameThread.quitSafely();
            try {
                frameThread.join(1000);
            } catch (InterruptedException e) {
                Log.e(TAG, "等待抽帧线程结束时被中断", e);
            }
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

        // 释放资源（videoFrameExtractor 已在上面交给抽帧线程释放）
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