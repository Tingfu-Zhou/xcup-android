package com.example.helloworld;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.projection.MediaProjectionManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.os.SystemClock;
import android.provider.Settings;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.android.material.appbar.MaterialToolbar;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

public class OnlineAnalysisActivity extends AppCompatActivity implements OnlineAnalysisService.OnlineDataCallback {
    private static final String TAG = "OnlineAnalysisActivity";
    private static final int REQUEST_CODE_MEDIA_PROJECTION = 1001;
    private static final int REQUEST_CODE_AUDIO_PERMISSION = 1003;

    // UI组件
    private TextView tvStatus;

    // 分析相关
    private VideoClassifierHelper videoClassifier;
    private AudioInferenceHelper audioHelper;

    // 线程控制
    private final AtomicBoolean isAnalysisPaused = new AtomicBoolean(true);
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

    // 模型结果：动作串 + 置信度 + 产出时刻作为一个对象原子发布，融合线程只读这一个对象
    // （融合仲裁按产出时刻去重，三者必须来自同一次推理）
    private final AtomicReference<FusionArbiter.Result> latestVideoResult = new AtomicReference<>(null);
    private final AtomicReference<FusionArbiter.Result> latestAudioResult = new AtomicReference<>(null);
    private final AtomicReference<String> latestBluetoothAction = new AtomicReference<>("");

    // 帧缓存环形缓冲：最近 3 秒、间隔 250ms 采样的 12 帧
    private final ArrayDeque<Bitmap> frameWindow = new ArrayDeque<>();
    // 截屏帧转模型输入时复用的像素缓冲（只在视频线程用）
    private int[] screenPixels;
    // [去黑边] 在截屏里找出视频画面（ScreenCropper），再与训练一样取居中正方形。以下几个只在视频线程（videoLock 里）用
    private final ScreenCropper screenCropper = new ScreenCropper();
    private int[] lastVideoRect;                     // 上一次找到的视频画面（变了才打日志）
    private int frameWidth = -1, frameHeight = -1;   // 帧窗口里截图的尺寸：转屏后截图尺寸变了，窗口里的旧帧作废
    private static final int FRAME_WINDOW_SIZE = VideoClassifierHelper.NUM_FRAMES;

    // 视频推理步长：每 500ms 推理一次（与离线模式 VideoProcessActivity 保持一致）
    private long lastVideoInferenceTime = 0;
    private static final long VIDEO_INFERENCE_INTERVAL = 500;

    // [分组概率决策] 下游只关心“转/不转”，在 oral+sex 合并后的概率上做判定，
    // 避免转场期“总证据充分但单类不过线”被误判为 unclear（与离线模式一致）。
    // P(do) = P(oral)+P(sex) >= DO 阈值 -> "do"；P(plot/noise) >= 阈值 -> "Noise"；其余 -> unclear
    private static final float VIDEO_DO_PROB_THRESHOLD = 0.65f;
    private static final float VIDEO_PLOT_PROB_THRESHOLD = 0.60f;
    private static final float AUDIO_DO_PROB_THRESHOLD = 0.55f;
    private static final float AUDIO_NOISE_PROB_THRESHOLD = 0.60f;

    // 音频缓冲 - 使用统一的PcmCircularBuffer
    private PcmCircularBuffer pcmBuffer;
    private long audioStartTime = 0;
    private boolean hasAudioData = false;

    // 帧缓冲
    private Bitmap latestFrame = null;
    private final Object frameLock = new Object();

    // [融合仲裁] 启动 / 维持 / 停止 分开判断（规则见 FusionArbiter，与离线模式共用）。2026-10-07 起替代原来的平滑投票
    // smoothedFusion 和启动仲裁 applyDoStartGate（都已删除）。只在融合线程（主线程）访问；静音重置 / 恢复分析时 reset。
    private final FusionArbiter fusionArbiter = new FusionArbiter();

    // 蓝牙发送状态管理器变量
    private static final long BLUETOOTH_MIN_DURATION = 2000;
    private static final long BLUETOOTH_SEND_INTERVAL = 1600;
    private long lastBluetoothSendTime = 0;
    private String currentBluetoothState = "";
    private long currentStateStartTime = 0;
    private String pendingBluetoothState = "";
    private long pendingStateStartTime = 0;

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
    private int levelBoostNow = 0;                  // 此刻加几档（没打开 / 没到点为 0）：融合 tick 里更新，长时间静音时清零

    // 主线程融合循环间隔
    // [延迟优化] 1000 -> 500：蓝牙实际发送频率仍由状态机节流（与离线模式一致）
    private static final long MAIN_FUSION_INTERVAL = 500;
    // 视频线程间隔（帧缓存间隔 250ms）
    private static final long VIDEO_LOOP_INTERVAL_MS = 250;
    //音频线程间隔
    private static final long AUDIO_LOOP_TICK_MS = 1000;

    // [12.30] Loudness 档位估计器（音频节律输出）
    private final AudioLoudnessLevelEstimator loudnessEstimator = new AudioLoudnessLevelEstimator(16000);

    // [12.30] 音频“响度档位”结果（线程安全共享）
    private final java.util.concurrent.atomic.AtomicInteger latestAudioLoudLevel =
            new java.util.concurrent.atomic.AtomicInteger(0);
    private final java.util.concurrent.atomic.AtomicReference<Float> latestAudioLoudConf =
            new java.util.concurrent.atomic.AtomicReference<>(0f);
    private final java.util.concurrent.atomic.AtomicLong latestAudioLoudTsMs =
            new java.util.concurrent.atomic.AtomicLong(0L);
    private final java.util.concurrent.atomic.AtomicBoolean latestAudioLoudValid =
            new java.util.concurrent.atomic.AtomicBoolean(false);



    // MediaProjection相关
    private MediaProjectionManager projectionManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_online_analysis);

        // 初始化UI组件
        tvStatus = findViewById(R.id.tvStatus);

        MaterialToolbar topAppBar = findViewById(R.id.topAppBar);
        if (topAppBar != null) {
            topAppBar.setNavigationOnClickListener(v -> finish());
        }

        // [定时加档] 进入页面时读一次手动电子菜单里的设置
        levelBoost = ManualControlActivity.loadLevelBoost(this);

        // 初始化推理助手
        videoClassifier = new VideoClassifierHelper(this);
        audioHelper = new AudioInferenceHelper(this);

        // 初始化音频缓冲 - 使用在线模式
        pcmBuffer = new PcmCircularBuffer(16000, 10, true);

        // 初始化主线程Handler
        mainHandler = new Handler(Looper.getMainLooper());

        // 获取MediaProjectionManager
        projectionManager = (MediaProjectionManager) getSystemService(Context.MEDIA_PROJECTION_SERVICE);

        // 设置服务回调
        OnlineAnalysisService.setDataCallback(this);

        // 检查并请求权限
        checkAndRequestPermissions();
    }

    private void checkAndRequestPermissions() {
        // ======= 修改：先检查音频录制权限 =======
        // 改进音频权限请求流程
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                != PackageManager.PERMISSION_GRANTED) {

            // 检查是否需要显示权限说明（用户之前拒绝过）
            if (ActivityCompat.shouldShowRequestPermissionRationale(this,
                    Manifest.permission.RECORD_AUDIO)) {
                // 显示解释弹窗
                new AlertDialog.Builder(this)
                        .setTitle(R.string.online_audio_permission_title)
                        .setMessage(R.string.online_audio_permission_rationale)
                        .setPositiveButton(R.string.online_grant_permission, (dialog, which) -> {
                            ActivityCompat.requestPermissions(this,
                                    new String[]{Manifest.permission.RECORD_AUDIO},
                                    REQUEST_CODE_AUDIO_PERMISSION);
                        })
                        .setNegativeButton(R.string.online_cancel, (dialog, which) -> finish())
                        .show();
            } else {
                // 第一次请求或者用户选择了"不再询问"
                // 先尝试直接请求
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.RECORD_AUDIO},
                        REQUEST_CODE_AUDIO_PERMISSION);
            }
            return;
        }

        // 请求屏幕录制权限
        requestScreenCapture();
    }

    private void requestScreenCapture() {
        Intent captureIntent = projectionManager.createScreenCaptureIntent();
        startActivityForResult(captureIntent, REQUEST_CODE_MEDIA_PROJECTION);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == REQUEST_CODE_MEDIA_PROJECTION) {
            if (resultCode == Activity.RESULT_OK && data != null) {
                // 启动服务
                Intent serviceIntent = new Intent(this, OnlineAnalysisService.class);
                serviceIntent.putExtra("resultCode", resultCode);
                serviceIntent.putExtra("data", data);

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    startForegroundService(serviceIntent);
                } else {
                    startService(serviceIntent);
                }

                // 启动多线程分析
                startMultiThreadAnalysis();

                tvStatus.setText(R.string.online_status_running);
            } else {
                Toast.makeText(this, R.string.online_screen_capture_required, Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == REQUEST_CODE_AUDIO_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // 音频权限授予，继续检查其他权限
                checkAndRequestPermissions();
            } else {
                // ======= 添加开始：处理权限被拒绝的情况 =======
                // 检查是否选择了"不再询问"
                if (!ActivityCompat.shouldShowRequestPermissionRationale(this,
                        Manifest.permission.RECORD_AUDIO)) {
                    // 用户选择了"不再询问"，引导去设置页面
                    new AlertDialog.Builder(this)
                            .setTitle(R.string.online_audio_permission_title)
                            .setMessage(R.string.online_audio_permission_denied)
                            .setPositiveButton(R.string.online_open_settings, (dialog, which) -> {
                                // 跳转到应用设置页面
                                Intent intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS);
                                Uri uri = Uri.fromParts("package", getPackageName(), null);
                                intent.setData(uri);
                                startActivity(intent);
                                finish();
                            })
                            .setNegativeButton(R.string.online_cancel, (dialog, which) -> finish())
                            .show();
                } else {
                    // 用户只是拒绝了，没有选择"不再询问"
                    Toast.makeText(this, R.string.online_audio_permission_required, Toast.LENGTH_LONG).show();
                    finish();
                }
                // ======= 添加结束 =======
            }
        }
    }
    @Override
    public void onFrameAvailable(Bitmap frame, long timestamp) {
        synchronized (frameLock) {
            // 不要立即回收旧的bitmap，让GC自动处理
            latestFrame = frame;
        }
    }

    @Override
    public void onAudioDataAvailable(byte[] audioData, int length) {
        // 将音频数据添加到缓冲区 - 使用统一的PcmCircularBuffer
        pcmBuffer.addByteData(audioData, length);
        hasAudioData = true;
    }

    @Override
    public void onAudioStateChanged(boolean hasAudio) {
        if (hasAudio) {
            Log.d(TAG, "[同步分析] 检测到音频，恢复分析");
            resumeAnalysis();
        } else {
            Log.d(TAG, "[同步分析] 音频静音，暂停分析");
            pauseAnalysis();
        }
        // 如果静音超过某个阈值（如5秒），才清空缓冲
        mainHandler.postDelayed(() -> {
            if (isAnalysisPaused.get()) {
                resetAnalysisBuffers();
            }
        }, 5000);
    }

    @Override
    public void onServiceStopped() {
        Log.d(TAG, "服务已停止");
        finish();
    }

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
                    Bitmap frame;
                    synchronized (frameLock) {
                        frame = latestFrame;
                        latestFrame = null;
                    }

                    if (frame != null && !frame.isRecycled()) {
                        cacheFrameAndInfer(frame);
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
     * 缓存一帧到滑动窗口；窗口满 12 帧且距上次推理 ≥ 1000ms 时执行一次推理。
     */
    private void cacheFrameAndInfer(Bitmap frame) {
        synchronized (videoLock) {
            int w = frame.getWidth(), h = frame.getHeight();
            // 截图尺寸变了（转屏后 OnlineAnalysisService 换了截图尺寸）：窗口里的旧帧是另一种画面布局，作废重攒
            if (w != frameWidth || h != frameHeight) {
                if (!frameWindow.isEmpty()) {
                    Log.i(TAG, String.format("[视频线程] 截图尺寸 %dx%d → %dx%d，清空帧窗口", frameWidth, frameHeight, w, h));
                }
                frameWindow.clear();
                frameWidth = w;
                frameHeight = h;
            }
            // 与训练一致：先去掉截屏里视频四周的黑边（ScreenCropper），再居中裁正方形、面积平均缩到 160（FramePreprocessor）。
            // 2026-10-14 之前是整屏直接压扁成 160×160；2026-10-19 之前不去黑边，竖屏全屏播放、截图没跟上转屏时，
            // 居中正方形里大半是黑边，视频被缩成中间一条
            if (screenPixels == null || screenPixels.length < w * h) {
                screenPixels = new int[w * h];
            }
            frame.getPixels(screenPixels, 0, w, 0, 0, w, h);
            int[] video = ScreenCropper.ENABLED ? screenCropper.update(screenPixels, w, h) : new int[]{0, 0, w, h};
            if (!Arrays.equals(video, lastVideoRect)) {
                Log.i(TAG, String.format("[视频线程] 截图 %dx%d 里的视频画面：(%d, %d) %dx%d",
                        w, h, video[0], video[1], video[2], video[3]));
                lastVideoRect = video;
            }
            Bitmap scaled = VideoClassifierHelper.toModelBitmap(screenPixels, w, h, FramePreprocessor.cropRect(video));
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
            Log.d(TAG, "[计时] MobileNetV3Small 推理耗时: "
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

        // 原子更新结果
        latestVideoResult.set(new FusionArbiter.Result(actionClass, confidence, System.currentTimeMillis()));

        String displayText = actionClass.isEmpty()
                ? "V: unclear"
                : String.format("V: %s (%.2f)", actionClass, confidence);
        Log.d(TAG, "[同步分析] [视频线程] " + displayText
                + (result != null ? " probs=" + java.util.Arrays.toString(result.probs) : ""));
    }

    private long lastAudioInferenceTime = 0;
    private static final long AUDIO_INFERENCE_INTERVAL = 1000;

    private void startAudioAnalysisLoop() {
        Runnable audioRunnable = new Runnable() {
            @Override
            public void run() {
                if (shouldStop.get()) {
                    Log.d(TAG, "[音频线程] 收到停止信号，退出循环");
                    return;
                }

                if (isAnalysisPaused.get() || !hasAudioData) {
                    audioHandler.postDelayed(this, 100);
                    return;
                }
                long t0 = System.currentTimeMillis();

                long currentSystemTime = System.currentTimeMillis();

                if ((currentSystemTime - lastAudioInferenceTime) >= AUDIO_INFERENCE_INTERVAL) {
                    synchronized (audioLock) {
                        // 读取2秒音频数据 - 使用统一的接口
                        float[] audioSegment = pcmBuffer.getLatestData(32000);

                        // **************[MOD] 音频 Loudness → 档位分析（替代频率估计）**************
                        float[] last1s = pcmBuffer.getLatestData(8000); // 0.5秒, 16 kHz/s
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
                            long tAudioInferStart = System.currentTimeMillis();
                            AudioInferenceHelper.AudioInferenceResult result = audioHelper.predict(audioSegment);
                            long tAudioInferEnd = System.currentTimeMillis();
                            //Log.d(TAG, "[计时] 音频推理耗时: " + (tAudioInferEnd - tAudioInferStart) + " ms");

                            // 音频模型类别: 0=sex(做爱), 1=oral(口交), 2=noise(杂音)
                            // [分组概率决策] P(do)=P(sex)+P(oral) >= 阈值 -> "do"(转);
                            // P(noise) >= 阈值 -> "Noise"(不转); 其余 -> ""(unclear)
                            String action;
                            float confidence;
                            if (result.index < 0 || result.probs.length < 3) {
                                action = "";
                                confidence = 0f;
                                Log.w(TAG, "[同步分析] [音频线程] 推理结果无效，判为 unclear");
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
                                    Log.d(TAG, String.format("[同步分析] [音频线程] P(do)=%.2f / P(noise)=%.2f 均未过阈值，判为 unclear",
                                            pDo, pNoise));
                                }
                            }

                            // [音频线索] YAMNet 自带的 AudioSet 分数判"这一刻音频靠不靠得住"，只给融合仲裁的启动判断用
                            // （规则见 AudioCue、FusionArbiter；与 VideoProcessActivity 相同）
                            int cue = AudioCue.of(result.audioSet);
                            latestAudioResult.set(new FusionArbiter.Result(action, confidence, System.currentTimeMillis(), cue));

                            final String displayText = (action.isEmpty()
                                    ? "A: unclear"
                                    : String.format("A: %s (%.2f)", action, confidence))
                                    + (cue == AudioCue.UNKNOWN ? "" : " · " + AudioCue.label(cue));
                            Log.d(TAG, "[同步分析] [音频线程] " + displayText);

                            lastAudioInferenceTime = currentSystemTime;
                        }
                    }
                }

                long elapsed = System.currentTimeMillis() - t0;
                long nextDelay = Math.max(0, AUDIO_LOOP_TICK_MS - elapsed);
                audioHandler.postDelayed(this, nextDelay);
            }
        };

        audioHandler.post(audioRunnable);
    }

    private void startMainFusionLoop() {
        Runnable fusionRunnable = new Runnable() {
            @Override
            public void run() {
                if (shouldStop.get()) {
                    return;
                }

                if (isAnalysisPaused.get()) {
                    levelBoost.pause(SystemClock.elapsedRealtime()); // [定时加档] 静音暂停期间不计时
                    mainHandler.postDelayed(this, MAIN_FUSION_INTERVAL);
                    return;
                }
                long t0 = System.currentTimeMillis();

                // [新增] 检查BLE是否被本地按键暂停
                if (BLEManager.globalManager != null && BLEManager.globalManager.isPausedByLocal()) {
                    Log.d(TAG, "[融合线程] BLE被本地按键暂停，跳过蓝牙发送");
                    // 仍然进行分析但不发送蓝牙命令
                }

                FusionArbiter.Result vRes = latestVideoResult.get();
                FusionArbiter.Result aRes = latestAudioResult.get();
                String videoAction = vRes == null ? "" : vRes.action;
                String audioAction = aRes == null ? "" : aRes.action;
                float videoConf = vRes == null ? 0f : vRes.conf;
                float audioConf = aRes == null ? 0f : aRes.conf;
                long videoTime = vRes == null ? 0L : vRes.tsMs;
                long audioTime = aRes == null ? 0L : aRes.tsMs;

                // 音频节奏
                boolean audioFreqValid = latestAudioLoudValid.get();
                float audioFreq = (float) latestAudioLoudLevel.get();    // 用 level 作为 freq 输入
                float audioFreqConf = latestAudioLoudConf.get();
                long audioFreqTs = latestAudioLoudTsMs.get();

                long currentTime = System.currentTimeMillis();
                long videoAge = videoTime > 0 ? currentTime - videoTime : Long.MAX_VALUE;
                long audioAge = audioTime > 0 ? currentTime - audioTime : Long.MAX_VALUE;
                long audioFreqAge = audioFreqTs > 0 ? currentTime - audioFreqTs : Long.MAX_VALUE;

                final long MAX_AGE = 2000;

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

                if (videoAge > MAX_AGE) {
                    videoAction = "";
                    videoConf = 0f;
                    Log.d(TAG, "[融合] 视频结果过期（" + videoAge + "ms），已忽略");
                }

                if (audioAge > MAX_AGE) {
                    audioAction = "";
                    audioConf = 0f;
                    Log.d(TAG, "[融合] 音频结果过期（" + audioAge + "ms），已忽略");
                }

                // 注：oral/sex 的归并已在各分析线程内完成（分组概率决策只输出 "do"/"Noise"/""），
                // 此处无需再做 oral->do 归一化。

                // [融合仲裁] 启动 / 维持 / 停止 分开判断，证据按"新推理结果"计数（结果时间戳去重）；
                // 音频线索与音频结果来自同一个 Result
                String finalAction = fusionArbiter.decide(currentTime,
                        videoAction, videoConf, videoTime > 0 ? videoTime : -1,
                        audioAction, audioConf, audioTime > 0 ? audioTime : -1,
                        aRes == null ? AudioCue.UNKNOWN : aRes.cue,
                        isSexAction(currentBluetoothState));
                Log.d(TAG, String.format("[融合仲裁] %s -> %s", fusionArbiter.lastReason(), finalAction));
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

                long elapsed = System.currentTimeMillis() - t0;
                long nextDelay = Math.max(0, MAIN_FUSION_INTERVAL - elapsed);
                mainHandler.postDelayed(this, nextDelay);
            }
        };

        mainHandler.post(fusionRunnable);
    }

    // audioFreq 实际承载的是 loudness level（float），这里做 0..8 钳制
    private static int clampLevelFromLoudness(final float levelLike) {
        if (Float.isNaN(levelLike)) return 0;
        int lv = Math.round(levelLike);
        if (lv < 0) lv = 0;
        if (lv > 8) lv = 8;
        return lv;
    }


    /**
     * 计算最终节律档位（0..8）。
     *
     * <p>策略：使用音频响度档位，并做“置信度阈值 + 方向性门控（涨档更严格，降档更宽松）”。</p>
     */
    private int computeFinalFreq(float audioFreq, float audioFreqConf) {
        int finalFreq = clampLevelFromLoudness(audioFreq); // audioFreq 实际是 loudness level(float)
        Log.d(TAG, "[音频响度] 得到音频档位: " + audioFreq + " 置信度: " + audioFreqConf + " 档位: " + finalFreq);

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
    private boolean currentStateSupportsSpeed() {                              // NEW
        return currentBluetoothState.startsWith("do");
    }

    // （可选）门控：newAction 是否属于“做爱大类”（你若已有类似函数，可直接替换）
    private boolean isSexAction(String action) {                               // NEW
        return action != null && action.startsWith("do");
    }

    // NEW: 迟滞门限辅助
    private int upThreshold(int cur)   { return Math.min(10, cur + 1); }
    private int downThreshold(int cur) { return Math.max(0,  cur - 1); }

    /** [定时加档] 融合循环在跑：接着计时，更新此刻该加几档。计时用单调时钟，改系统时间不影响 */
    private void updateLevelBoost() {
        long now = SystemClock.elapsedRealtime();
        levelBoost.resume(now);
        levelBoostNow = levelBoost.boostAt(now);
    }

    /** [定时加档] 长时间静音（当作视频播完）：从头计时，之后发出去的档位不再加 */
    private void resetLevelBoost() {
        levelBoost.reset();
        levelBoostNow = 0;
    }

    /** [定时加档] 分析档位 → 发给设备的档位：到点后加 levelBoostNow 档（0 档不加，最高 BLEManager.LEVEL_MAX） */
    private int boostLevel(int level) {
        return LevelBoost.apply(level, levelBoostNow, BLEManager.LEVEL_MAX);
    }

    private void pauseAnalysis() {
        isAnalysisPaused.set(true);
        // 在线模式只在静音时暂停，而 OnlineAnalysisService 判定静音的同时已经直接给设备发了 Noise。
        // 这里把蓝牙状态同步成 Noise：否则恢复后 app 以为马达还在转，维持时重申的 "do" 不会真正发出去，
        // 马达就一直停着（表现为声音回来了杯子却不转）。
        // 服务是在音频线程里发的 Noise，主线程这边可能紧接着又发出一个 "do"（比如档位重发），
        // 所以这里在主线程再补发一次 Noise（重复发没有副作用），保证设备上最后一条是 Noise。
        if (isSexAction(currentBluetoothState)) {
            if (BLEManager.globalManager != null && BLEManager.globalManager.isConnected()
                    && !BLEManager.globalManager.isPausedByLocal()) {
                BLEManager.globalManager.sendAction("Noise", 0);
            }
            long now = System.currentTimeMillis();
            currentBluetoothState = "Noise";
            currentStateStartTime = now;
            pendingBluetoothState = "Noise";
            pendingStateStartTime = now;
            lastBluetoothSendTime = now;
            lastSentLevel = 0;
            latestBluetoothAction.set("Noise");
        }
        Log.i(TAG, "暂停分析");
    }

    private void resumeAnalysis() {
        isAnalysisPaused.set(false);
        // [融合仲裁] 暂停前的证据已经过时，清空重来。在线模式只在静音时暂停，恢复后的头一两个结果的模型窗口里
        // 还混着静音，一并丢掉（音频兜底启动要求窗口内全是"转"，一个陈旧的"不转"就会把启动拖后 9 秒）。
        fusionArbiter.reset(System.currentTimeMillis(), FusionArbiter.SILENCE_RESUME_WARMUP_MS);
        Log.i(TAG, "恢复分析");
    }

    private void resetAnalysisBuffers() {
        // 清空帧缓存和 pcm 缓冲，避免读取到“前一个时间点”的旧数据
        synchronized (videoLock) {
            frameWindow.clear();
            lastVideoInferenceTime = 0;
            // 静音这么久，多半换了视频或应用：黑边从头找
            screenCropper.reset();
            lastVideoRect = null;
        }

        synchronized (audioLock) {
            pcmBuffer.reset();
        }

        // [融合仲裁] 清空证据：重置之前产出的结果一律不算
        fusionArbiter.reset(System.currentTimeMillis());

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

        // [12.20] reset loudness estimator
        loudnessEstimator.reset();
        // [12.30] 清空 loudness 输出
        latestAudioLoudLevel.set(0);
        latestAudioLoudConf.set(0f);
        latestAudioLoudTsMs.set(0L);
        latestAudioLoudValid.set(false);

        // [定时加档] 在线模式看不到视频什么时候播完：静音久到要清空分析缓冲（约 10 秒），就当这个视频播完了，从头计时
        resetLevelBoost();

        Log.d(TAG, "[在线分析] [同步分析] 已重置所有分析缓冲");
    }

    @Override
    protected void onDestroy() {
        shouldStop.set(true);

        // 停止服务
        Intent serviceIntent = new Intent(this, OnlineAnalysisService.class);
        stopService(serviceIntent);

        // 停止所有线程
        if (videoThread != null) {
            videoThread.quitSafely();
        }

        if (audioThread != null) {
            audioThread.quitSafely();
        }

        if (mainHandler != null) {
            mainHandler.removeCallbacksAndMessages(null);
        }

        if (videoClassifier != null) {
            videoClassifier.close();
        }

        if (audioHelper != null) {
            audioHelper.close();
        }
        super.onDestroy();
    }
}