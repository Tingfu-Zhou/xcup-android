package com.example.helloworld;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.appbar.MaterialToolbar;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.button.MaterialButtonToggleGroup;
import com.google.android.material.card.MaterialCardView;
import com.google.android.material.slider.Slider;

/**
 * 手动电子菜单：
 *  - 不依赖视频分析，用户直接在 app 上选择变频模式（PATTERN_ID 1..3）与马达强度（INT_LEVEL 0..10）
 *  - 每次调节实时下发 SetPattern(0x04)，等价于 buildSetPatternFrame(PATTERN, LEVEL, 0, 1)
 *  - 提供 StopAll(0x02) 紧急停止；设备被本地按键锁定时提示并支持 ResumeAppControl(0x12)
 *  - 顶部返回按钮 / 系统返回键均可回到主页面
 *  - 这里选择的变频模式同时作为「视频分析模式」的变频模式：分析链路只决定转/不转与强度，
 *    转动方式由本页决定（见 BLEManager#setAnalysisPattern）
 */
public class ManualControlActivity extends AppCompatActivity {

    private static final String PREFS = "manual_control";
    private static final String KEY_PATTERN = "pattern";

    /** 拖动滑杆时的最小下发间隔，避免刷屏式写特征值把 BLE 链路打满 */
    private static final long SEND_MIN_INTERVAL_MS = 150L;

    /** 协议 §9.2 强度档位对应的转速（RPM），下标 = 档位 0..10 */
    private static final int[] LEVEL_RPM = {0, 190, 220, 240, 270, 280, 290, 295, 300, 310, 320};
    /** 协议 §9.2 强度档位对应的伸缩频率（Hz），下标 = 档位 0..10 */
    private static final String[] LEVEL_HZ = {
            "0", "1.06", "1.22", "1.33", "1.50", "1.56", "1.61", "1.64", "1.67", "1.72", "1.78"
    };

    private MaterialButtonToggleGroup togglePattern;
    private TextView tvPatternDesc;
    private Slider sliderLevel;
    private TextView tvLevelValue;
    private TextView tvLevelDetail;
    private TextView tvConnStatus;
    private TextView tvSendStatus;
    private MaterialCardView cardPaused;
    private MaterialButton btnStopAll;

    private final Handler ui = new Handler(Looper.getMainLooper());

    private int currentPattern = BLEManager.PATTERN_MIN;
    private int currentLevel = BLEManager.LEVEL_MIN;

    private long lastSendAt = 0L;
    private Runnable pendingSend;

    /** 页面自身的连接监听，避免覆盖 MainActivity 通过 setConnectionCallback 注册的主回调 */
    private final BLEManager.ConnectionCallback connectionListener = new BLEManager.ConnectionCallback() {
        @Override
        public void onConnectionStateChanged(boolean connected) {
            refreshConnectionUi();
        }

        @Override
        public void onScanStarted() {
            // 手动页面不发起扫描，无需处理
        }

        @Override
        public void onScanFailed(String reason) {
            // 手动页面不发起扫描，无需处理
        }
    };

    private final BLEManager.PauseStateCallback pauseListener = paused -> refreshPausedUi(paused);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_manual_control);

        MaterialToolbar toolbar = findViewById(R.id.topAppBar);
        toolbar.setNavigationOnClickListener(v -> finish());

        togglePattern = findViewById(R.id.togglePattern);
        tvPatternDesc = findViewById(R.id.tvPatternDesc);
        sliderLevel = findViewById(R.id.sliderLevel);
        tvLevelValue = findViewById(R.id.tvLevelValue);
        tvLevelDetail = findViewById(R.id.tvLevelDetail);
        tvConnStatus = findViewById(R.id.tvConnStatus);
        tvSendStatus = findViewById(R.id.tvSendStatus);
        cardPaused = findViewById(R.id.cardPaused);
        btnStopAll = findViewById(R.id.btnStopAll);

        // 恢复上次选择的模式；强度一律从 0（停止）开始，避免进页面就误动
        SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        currentPattern = BLEManager.clampPattern(prefs.getInt(KEY_PATTERN, BLEManager.PATTERN_MIN));
        currentLevel = BLEManager.LEVEL_MIN;

        BLEManager bleOnCreate = BLEManager.globalManager;
        if (bleOnCreate != null) {
            bleOnCreate.setAnalysisPattern(currentPattern);
        }

        togglePattern.check(patternToButtonId(currentPattern));
        tvPatternDesc.setText(patternDescRes(currentPattern));
        sliderLevel.setValue(currentLevel);
        updateLevelText(currentLevel);

        togglePattern.addOnButtonCheckedListener((group, checkedId, isChecked) -> {
            if (!isChecked) return;
            currentPattern = buttonIdToPattern(checkedId);
            tvPatternDesc.setText(patternDescRes(currentPattern));
            getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                    .putInt(KEY_PATTERN, currentPattern).apply();
            // 同步给视频分析模式：分析链路下一次发送即用新模式，
            // 若此刻分析正驱动马达转动，BLEManager 会立即补发一帧
            BLEManager ble = BLEManager.globalManager;
            if (ble != null) {
                ble.setAnalysisPattern(currentPattern);
            }
            // 停止档位下切模式只更新界面，不唤醒马达
            if (currentLevel > BLEManager.LEVEL_MIN) {
                sendNow();
            }
        });

        sliderLevel.addOnChangeListener((slider, value, fromUser) -> {
            currentLevel = Math.round(value);
            updateLevelText(currentLevel);
            if (fromUser) {
                scheduleSend();
            }
        });

        // 松手时补发一次最终值，保证界面与设备一致
        sliderLevel.addOnSliderTouchListener(new Slider.OnSliderTouchListener() {
            @Override
            public void onStartTrackingTouch(@NonNull Slider slider) {
                // no-op
            }

            @Override
            public void onStopTrackingTouch(@NonNull Slider slider) {
                cancelPendingSend();
                sendNow();
            }
        });

        btnStopAll.setOnClickListener(v -> {
            cancelPendingSend();
            BLEManager ble = BLEManager.globalManager;
            if (ble == null || !ble.isConnected()) {
                toast(R.string.manual_toast_disconnected);
                return;
            }
            // StopAll 优先级最高，锁定期内设备也必须接受
            if (ble.sendStopAll()) {
                currentLevel = BLEManager.LEVEL_MIN;
                sliderLevel.setValue(currentLevel);
                updateLevelText(currentLevel);
                tvSendStatus.setText(R.string.manual_status_sent_stop);
            } else {
                tvSendStatus.setText(R.string.manual_status_send_failed);
            }
        });

        findViewById(R.id.btnResume).setOnClickListener(v -> {
            BLEManager ble = BLEManager.globalManager;
            if (ble != null && ble.isPausedByLocal()) {
                ble.sendResumeControl();
                toast(R.string.manual_toast_resuming);
            }
        });

        BLEManager ble = BLEManager.globalManager;
        if (ble != null) {
            ble.addConnectionListener(connectionListener);
            ble.addPauseStateListener(pauseListener);
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        refreshConnectionUi();
        BLEManager ble = BLEManager.globalManager;
        refreshPausedUi(ble != null && ble.isPausedByLocal());
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cancelPendingSend();
        BLEManager ble = BLEManager.globalManager;
        if (ble != null) {
            ble.removeConnectionListener(connectionListener);
            ble.removePauseStateListener(pauseListener);
        }
    }

    // ====== 下发节流 ======

    /** 拖动过程中限流下发：距上次下发不足 SEND_MIN_INTERVAL_MS 时排队一次延迟发送 */
    private void scheduleSend() {
        long now = SystemClock.uptimeMillis();
        long wait = SEND_MIN_INTERVAL_MS - (now - lastSendAt);
        if (wait <= 0) {
            sendNow();
            return;
        }
        if (pendingSend != null) return; // 已有排队任务，落地时会读取最新档位
        pendingSend = () -> {
            pendingSend = null;
            sendNow();
        };
        ui.postDelayed(pendingSend, wait);
    }

    private void cancelPendingSend() {
        if (pendingSend != null) {
            ui.removeCallbacks(pendingSend);
            pendingSend = null;
        }
    }

    /** 立即下发当前「模式 + 强度」 */
    private void sendNow() {
        lastSendAt = SystemClock.uptimeMillis();

        BLEManager ble = BLEManager.globalManager;
        if (ble == null || !ble.isConnected()) {
            tvSendStatus.setText(R.string.manual_status_disconnected);
            toast(R.string.manual_toast_disconnected);
            return;
        }
        if (ble.isPausedByLocal()) {
            tvSendStatus.setText(R.string.manual_paused_title);
            toast(R.string.manual_toast_paused);
            return;
        }

        if (ble.sendManualPattern(currentPattern, currentLevel)) {
            tvSendStatus.setText(getString(R.string.manual_status_sent, currentPattern, currentLevel));
        } else {
            tvSendStatus.setText(R.string.manual_status_send_failed);
        }
    }

    // ====== 界面刷新 ======

    private void updateLevelText(int level) {
        tvLevelValue.setText(String.valueOf(level));
        if (level <= 0) {
            tvLevelDetail.setText(R.string.manual_level_detail_stop);
        } else {
            tvLevelDetail.setText(getString(R.string.manual_level_detail,
                    LEVEL_RPM[level], LEVEL_HZ[level]));
        }
    }

    private void refreshConnectionUi() {
        BLEManager ble = BLEManager.globalManager;
        boolean connected = ble != null && ble.isConnected();

        tvConnStatus.setText(connected
                ? R.string.manual_status_ready
                : R.string.manual_status_disconnected);

        if (!connected) {
            cancelPendingSend();
            tvSendStatus.setText(R.string.manual_status_disconnected);
        }
        updateControlsEnabled();
    }

    private void refreshPausedUi(boolean paused) {
        cardPaused.setVisibility(paused ? View.VISIBLE : View.GONE);
        if (paused) {
            cancelPendingSend();
            tvSendStatus.setText(R.string.manual_paused_title);
        } else {
            BLEManager ble = BLEManager.globalManager;
            if (ble != null && ble.isConnected()) {
                tvSendStatus.setText(R.string.manual_status_ready);
            }
        }
        updateControlsEnabled();
    }

    /**
     * 未连接、或设备被本地按键锁定时，模式/强度控件置灰——锁定期内设备只会回 BUSY，
     * 让控件可拖动只会产生无效下发。紧急停止按钮不受锁定影响（StopAll 优先级最高）。
     */
    private void updateControlsEnabled() {
        BLEManager ble = BLEManager.globalManager;
        boolean connected = ble != null && ble.isConnected();
        boolean paused = ble != null && ble.isPausedByLocal();
        boolean canControl = connected && !paused;

        togglePattern.setEnabled(canControl);
        for (int i = 0; i < togglePattern.getChildCount(); i++) {
            togglePattern.getChildAt(i).setEnabled(canControl);
        }
        sliderLevel.setEnabled(canControl);
        btnStopAll.setEnabled(connected);
    }

    // ====== 模式与按钮 ID 映射 ======

    private static int patternToButtonId(int pattern) {
        switch (pattern) {
            case 2:
                return R.id.btnPattern2;
            case 3:
                return R.id.btnPattern3;
            default:
                return R.id.btnPattern1;
        }
    }

    private static int buttonIdToPattern(int buttonId) {
        if (buttonId == R.id.btnPattern2) return 2;
        if (buttonId == R.id.btnPattern3) return 3;
        return 1;
    }

    private static int patternDescRes(int pattern) {
        switch (pattern) {
            case 2:
                return R.string.manual_pattern_2_desc;
            case 3:
                return R.string.manual_pattern_3_desc;
            default:
                return R.string.manual_pattern_1_desc;
        }
    }

    private void toast(int resId) {
        Toast.makeText(this, resId, Toast.LENGTH_SHORT).show();
    }
}
