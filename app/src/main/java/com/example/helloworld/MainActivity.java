package com.example.helloworld;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import androidx.appcompat.app.AlertDialog;
import android.widget.Toast;
import android.net.ConnectivityManager;
import android.net.NetworkCapabilities;
import android.net.NetworkInfo;
import android.content.Context;
import android.os.Build;
// [新增] BLE权限相关
import android.Manifest;
import android.content.pm.PackageManager;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.core.content.ContextCompat;
import java.util.ArrayList;
import java.util.List;
import android.os.Handler;
import android.os.Looper;

public class MainActivity extends AppCompatActivity {
    private View btnSelectVideo; // [MD3] 改为 MaterialCardView，用 View 引用
    private Button btnConnectBluetooth; // [修改] 改为真实BLE连接按钮
    private Button btnResumeControl;    // [新增] 恢复控制按钮
    private TextView tvBluetoothStatus; // [MD3] 蓝牙状态副文本
    private static final int REQUEST_CODE_SELECT_VIDEO = 101;
    private boolean updateDialogShown = false;

    // 隐私政策：Google Play 要求商店后台和 app 里都要有链接。和 iOS 版共用同一个页面
    private static final String PRIVACY_POLICY_URL = "https://download.xswy.tech/privacy.html";
    // 首次启动的 18+ 确认，确认过一次就不再弹
    private static final String PREFS = "app";
    private static final String KEY_AGE_CONFIRMED = "age_confirmed";

    // [新增] BLE管理器
    private BLEManager bleManager;

    // [新增] 权限请求Launcher
    private final ActivityResultLauncher<String[]> permissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestMultiplePermissions(), result -> {
                boolean allGranted = true;
                for (Boolean granted : result.values()) {
                    if (!granted) {
                        allGranted = false;
                        break;
                    }
                }
                if (allGranted) {
                    // 权限获取成功后开始扫描
                    startBLEScan();
                } else {
                    Toast.makeText(this, R.string.bluetooth_permission_required, Toast.LENGTH_SHORT).show();
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        SystemBarInsets.padContentRoot(this);

        btnSelectVideo = findViewById(R.id.btnSelectVideo);
        btnConnectBluetooth = findViewById(R.id.btnConnectBluetooth);
        btnResumeControl = findViewById(R.id.btnResumeControl); // [新增] 恢复控制按钮
        tvBluetoothStatus = findViewById(R.id.tvBluetoothStatus); // [MD3] 蓝牙状态副文本

        // [新增] 初始化BLE管理器
        bleManager = new BLEManager(this);
        BLEManager.globalManager = bleManager; // 设置全局实例

        // [新增] 设置BLE连接回调
        bleManager.setConnectionCallback(new BLEManager.ConnectionCallback() {
            @Override
            public void onConnectionStateChanged(boolean connected) {
                runOnUiThread(() -> {
                    if (connected) {
                        btnConnectBluetooth.setText(R.string.disconnect_bluetooth);
                        btnConnectBluetooth.setEnabled(true);
                        tvBluetoothStatus.setText(R.string.bluetooth_status_connected);
                        Toast.makeText(MainActivity.this, R.string.bluetooth_toast_connected, Toast.LENGTH_SHORT).show();
                    } else {
                        btnConnectBluetooth.setText(R.string.connect_bluetooth);
                        btnConnectBluetooth.setEnabled(true);
                        tvBluetoothStatus.setText(R.string.bluetooth_status_disconnected);
                        Toast.makeText(MainActivity.this, R.string.bluetooth_toast_disconnected, Toast.LENGTH_SHORT).show();
                    }
                });
            }

            @Override
            public void onScanStarted() {
                runOnUiThread(() -> {
                    btnConnectBluetooth.setText(R.string.bluetooth_button_scanning);
                    btnConnectBluetooth.setEnabled(false);
                    tvBluetoothStatus.setText(R.string.bluetooth_status_scanning);
                });
            }

            @Override
            public void onScanFailed(String reason) {
                runOnUiThread(() -> {
                    btnConnectBluetooth.setText(R.string.connect_bluetooth);
                    btnConnectBluetooth.setEnabled(true);
                    tvBluetoothStatus.setText(R.string.bluetooth_status_disconnected);
                    Toast.makeText(MainActivity.this, reason, Toast.LENGTH_LONG).show();
                });
            }
        });

        // [新增] 设置暂停状态回调
        bleManager.setPauseStateCallback(paused -> {
            runOnUiThread(() -> {
                if (paused) {
                    btnResumeControl.setEnabled(true);
                    btnResumeControl.setText(R.string.resume_control_paused);
                    btnResumeControl.setAlpha(1.0f); // 完全不透明
                    Toast.makeText(this, R.string.toast_device_local_control, Toast.LENGTH_LONG).show();
                } else {
                    btnResumeControl.setEnabled(false);
                    btnResumeControl.setText(R.string.resume_control_resumed);
                    btnResumeControl.setAlpha(0.5f); // 半透明（灰色效果）
                    // 延迟后恢复默认文本
                    new Handler(Looper.getMainLooper()).postDelayed(() -> {
                        btnResumeControl.setText(R.string.resume_control_inactive);
                    }, 2000);
                }
            });
        });

        // [新增] 初始化恢复控制按钮状态
        btnResumeControl.setEnabled(false);
        btnResumeControl.setAlpha(0.5f);
        btnResumeControl.setText(R.string.resume_control_inactive);

        // 选择手机相册中的视频
        btnSelectVideo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                intent.setType("video/*");
                intent.addCategory(Intent.CATEGORY_OPENABLE);
                startActivityForResult(intent, REQUEST_CODE_SELECT_VIDEO);
            }
        });

        // [修改] 蓝牙连接按钮 - 改为真实BLE扫描连接
        btnConnectBluetooth.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!bleManager.isConnected()) {
                    // 检查蓝牙是否可用
                    if (!bleManager.isBluetoothAvailable()) {
                        Toast.makeText(MainActivity.this, R.string.bluetooth_enable_prompt, Toast.LENGTH_SHORT).show();
                        return;
                    }
                    // 请求权限并开始扫描
                    requestBLEPermissions();
                } else {
                    // 断开连接
                    bleManager.disconnect();
                    btnConnectBluetooth.setText(R.string.connect_bluetooth);
                    tvBluetoothStatus.setText(R.string.bluetooth_status_disconnected);
                }
            }
        });

        // [新增] 恢复控制按钮点击事件
        btnResumeControl.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (bleManager.isPausedByLocal()) {
                    bleManager.sendResumeControl();
                    Toast.makeText(MainActivity.this, R.string.manual_toast_resuming, Toast.LENGTH_SHORT).show();
                }
            }
        });

        View btnOnlineAnalysis = findViewById(R.id.btnOnlineAnalysis); // [MD3] MaterialCardView
        btnOnlineAnalysis.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, OnlineAnalysisActivity.class);
                startActivity(intent);
            }
        });

        // [新增] 手动电子菜单入口：脱离视频分析，直接在 app 上选模式与强度
        View btnManualControl = findViewById(R.id.btnManualControl);
        btnManualControl.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, ManualControlActivity.class);
                startActivity(intent);
            }
        });

        findViewById(R.id.btnPrivacyPolicy).setOnClickListener(v -> openPrivacyPolicy());

        showAgeConfirmationIfNeeded();
    }

    // 首次启动：确认已满 18 周岁（不可取消；点「退出」直接关掉 app）
    private void showAgeConfirmationIfNeeded() {
        SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        if (prefs.getBoolean(KEY_AGE_CONFIRMED, false)) return;
        AlertDialog dialog = new AlertDialog.Builder(this)
                .setTitle(R.string.age_gate_title)
                .setMessage(R.string.age_gate_message)
                .setCancelable(false)
                .setPositiveButton(R.string.age_gate_confirm,
                        (d, w) -> prefs.edit().putBoolean(KEY_AGE_CONFIRMED, true).apply())
                .setNegativeButton(R.string.age_gate_exit, (d, w) -> finishAffinity())
                .setNeutralButton(R.string.privacy_policy, null)
                .create();
        dialog.show();
        // 点「隐私政策」不关弹窗：看完回来还要确认年龄
        dialog.getButton(AlertDialog.BUTTON_NEUTRAL).setOnClickListener(v -> openPrivacyPolicy());
    }

    private void openPrivacyPolicy() {
        try {
            startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(PRIVACY_POLICY_URL)));
        } catch (Exception e) {
            Toast.makeText(this, R.string.update_open_browser_failed, Toast.LENGTH_LONG).show();
        }
    }

    // [新增] 请求BLE权限
    private void requestBLEPermissions() {
        List<String> requiredPermissions = new ArrayList<>();

        if (Build.VERSION.SDK_INT >= 31) {
            // Android 12+需要的权限
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_SCAN)
                    != PackageManager.PERMISSION_GRANTED) {
                requiredPermissions.add(Manifest.permission.BLUETOOTH_SCAN);
            }
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_CONNECT)
                    != PackageManager.PERMISSION_GRANTED) {
                requiredPermissions.add(Manifest.permission.BLUETOOTH_CONNECT);
            }
        } else {
            // Android 12以下需要位置权限
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
                    != PackageManager.PERMISSION_GRANTED) {
                requiredPermissions.add(Manifest.permission.ACCESS_FINE_LOCATION);
            }
        }

        if (!requiredPermissions.isEmpty()) {
            permissionLauncher.launch(requiredPermissions.toArray(new String[0]));
        } else {
            // 权限已获取，开始扫描
            startBLEScan();
        }
    }

    // [新增] 开始BLE扫描
    private void startBLEScan() {
        bleManager.startScanAndConnect();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == REQUEST_CODE_SELECT_VIDEO && resultCode == RESULT_OK && data != null) {
            Uri selectedVideoUri = data.getData();
            if (selectedVideoUri != null) {
                // 启动 VideoProcessActivity，并传递选中视频的 Uri
                Intent intent = new Intent(MainActivity.this, VideoProcessActivity.class);
                intent.setData(selectedVideoUri);
                intent.putExtra("IS_ASSET", false); // 标识使用用户选中的视频
                intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
                startActivity(intent);
            }
        }
    }

    // 判断是否有可用网络（保持原有方法）
    private boolean hasNetwork() {
        ConnectivityManager cm = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
        if (cm == null) return false;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            android.net.Network network = cm.getActiveNetwork();
            if (network == null) return false;
            NetworkCapabilities nc = cm.getNetworkCapabilities(network);
            return nc != null && (nc.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)
                    || nc.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR)
                    || nc.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET));
        } else {
            NetworkInfo ni = cm.getActiveNetworkInfo();
            return ni != null && ni.isConnected();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        // [新增] 更新连接按钮状态
        if (bleManager != null && bleManager.isConnected()) {
            btnConnectBluetooth.setText(R.string.disconnect_bluetooth);
            tvBluetoothStatus.setText(R.string.bluetooth_status_connected);
        } else {
            btnConnectBluetooth.setText(R.string.connect_bluetooth);
            tvBluetoothStatus.setText(R.string.bluetooth_status_disconnected);
        }

        // 版本检查逻辑（保持不变）
        if (!hasNetwork()) {
            return;
        }
        if (updateDialogShown) return;
        UpdateChecker.check(this, info -> {
            if (info == null) return;
            runOnUiThread(() -> {
                updateDialogShown = true;
                showUpdateDialog(info);
            });
        });
    }

    // 显示更新对话框（保持原有方法）
    private void showUpdateDialog(UpdateChecker.UpdateInfo info) {
        String title = info.force
                ? getString(R.string.update_title_required)
                : getString(R.string.update_title_available, info.latestName == null ? "" : info.latestName);
        String msg = (info.notes == null ? "" : info.notes);

        AlertDialog.Builder b = new AlertDialog.Builder(this)
                .setTitle(title)
                .setMessage(msg)
                .setPositiveButton(R.string.update_download, (d, w) -> openDownloadPage(info));

        if (info.force) {
            b.setCancelable(false); // 强更：不可取消
        } else {
            b.setNegativeButton(R.string.update_later, null);
        }
        b.show();
    }

    // 打开下载页面（保持原有方法）
    private void openDownloadPage(UpdateChecker.UpdateInfo info) {
        String url = info.landingUrl;
        if (url == null || url.isEmpty()) {
            Toast.makeText(this, R.string.update_url_unavailable, Toast.LENGTH_LONG).show();
            return;
        }
        try {
            Uri u = Uri.parse(url);
            startActivity(new android.content.Intent(android.content.Intent.ACTION_VIEW, u));
        } catch (Exception e) {
            Toast.makeText(this, R.string.update_open_browser_failed, Toast.LENGTH_LONG).show();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // [新增] 清理BLE资源
        if (bleManager != null) {
            bleManager.close();
        }
    }
}