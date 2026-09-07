package com.example.helloworld;

import android.Manifest;
import android.annotation.SuppressLint;
import android.bluetooth.*;
import android.bluetooth.le.*;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.*;
import android.util.Log;
import androidx.annotation.RequiresPermission;
import androidx.core.content.ContextCompat;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * BLE通信管理器 - 替代原BluetoothHelper
 * 实现真实的BLE通信协议
 */
public class BLEManager {
    private static final String TAG = "BLEManager";

    // ====== BLE协议 UUIDs (与固件一致) ======
    private static final UUID SERVICE_UUID = UUID.fromString("e43c4cbf-9e30-44cc-b8ea-83561908a4e5");
    private static final UUID RX_CHAR_UUID = UUID.fromString("71cd6e15-8ed6-4727-b306-42a0e20fe7b6"); // App->Dev
    private static final UUID TX_CHAR_UUID = UUID.fromString("2cbb355f-d59a-4be6-aaab-fcfe27abcec4"); // Dev->App
    private static final UUID CCCD_UUID = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb");

    // ====== 目标设备标识 ======
    private static final String TARGET_NAME_PREFIX = "XCUP-A1B2"; // TODO: 根据实际设备名称修改

    // ====== 协议版本和命令定义 ======
    private static final int VER = 0x01;
    private static final int CMD_SET_PATTERN = 0x04;
    private static final int CMD_STOP_ALL = 0x02;
    private static final int CMD_QUERY_STATE = 0x03;
    private static final int CMD_STATE_RPT = 0x83;
    private static final int CMD_HEARTBEAT = 0x06;
    private static final int CMD_RESUME_APP = 0x12;

    // ====== StateReport扩展解析用常量 ======
    private static final int SRC_FW = 0, SRC_APP = 1, SRC_BUTTON = 2, SRC_SAFETY = 3;
    private static final int OWNER_IDLE = 0, OWNER_APP = 1, OWNER_LOCAL = 2;
    private static final int HOLD_NONE = 0, HOLD_TIMED = 1, HOLD_MANUAL = 2;

    // ====== 模式和强度定义 ======
    private static final int PATTERN_1 = 1; // 模式1：恒定
    private static final int PATTERN_2 = 2; // 模式2：脉冲
    private static final int PATTERN_3 = 3; // 模式3：波形
    private static int LEVEL = 0;

    // ====== [新增] 手动电子菜单可用范围（对应协议 §9.1 / §9.2）======
    public static final int PATTERN_MIN = 1;   // 变频模式最小值
    public static final int PATTERN_MAX = 3;   // 变频模式最大值
    public static final int LEVEL_MIN = 0;     // 强度档位最小值（0=停止）
    public static final int LEVEL_MAX = 10;    // 强度档位最大值
    //private static final int LEVEL_L = 1;    // 低
    //private static final int LEVEL_M = 2;    // 中
    //private static final int LEVEL_H = 3;    // 高

    // ====== [新增] 视频分析模式使用的变频模式 ======
    // 视频分析链路只决定「转 / 不转 + 强度档位」，「怎么转」由用户在手动电子菜单里选择，
    // 与手动页共用同一份 SharedPreferences，两个页面语义一致，用户不用学两套。
    private static final String PREFS_MANUAL = "manual_control";
    private static final String KEY_PATTERN = "pattern";
    private volatile int analysisPattern = PATTERN_1;
    /** 最近一次 sendAction（视频分析链路）下发的是否为「转」；手动下发 / StopAll / 断开会清零 */
    private volatile boolean analysisRunning = false;

    // ====== 全局单例 ======
    public static BLEManager globalManager = null;

    // ====== BLE组件 ======
    private Context context;
    private BluetoothAdapter adapter;
    private BluetoothLeScanner scanner;
    private BluetoothGatt gatt;
    private BluetoothGattCharacteristic rxChar, txChar;

    // ====== 连接状态 ======
    private volatile boolean isConnected = false;
    private volatile boolean isNotifying = false;
    private volatile boolean pausedByLocal = false; // 本地按键暂停标志

    // ====== 回调接口 ======
    // connectionCallback / pauseStateCallback 由主页面持有（保持原有语义）；
    // [新增] 额外的监听器列表，供手动控制页等其它界面临时挂载，互不覆盖。
    private ConnectionCallback connectionCallback;
    private PauseStateCallback pauseStateCallback;
    private final List<ConnectionCallback> connectionListeners = new CopyOnWriteArrayList<>();
    private final List<PauseStateCallback> pauseStateListeners = new CopyOnWriteArrayList<>();

    // ====== 序列号 ======
    private byte seq = 0;

    // ====== 扫描超时Handler ======
    private Handler scanTimeoutHandler = new Handler(Looper.getMainLooper());
    private Runnable scanTimeoutRunnable;

    /**
     * 连接状态回调接口
     */
    public interface ConnectionCallback {
        void onConnectionStateChanged(boolean connected);
        void onScanStarted();
        void onScanFailed(String reason);
    }

    /**
     * 暂停状态回调接口
     */
    public interface PauseStateCallback {
        void onPauseStateChanged(boolean paused);
    }

    /**
     * 构造函数
     */
    public BLEManager(Context context) {
        this.context = context.getApplicationContext();
        // 复用手动电子菜单里保存的模式，作为视频分析模式的变频模式
        analysisPattern = clampPattern(this.context
                .getSharedPreferences(PREFS_MANUAL, Context.MODE_PRIVATE)
                .getInt(KEY_PATTERN, PATTERN_1));
        BluetoothManager mgr = (BluetoothManager) context.getSystemService(Context.BLUETOOTH_SERVICE);
        adapter = mgr != null ? mgr.getAdapter() : null;
        scanner = adapter != null ? adapter.getBluetoothLeScanner() : null;
    }

    /**
     * 设置连接回调
     */
    public void setConnectionCallback(ConnectionCallback callback) {
        this.connectionCallback = callback;
    }

    /**
     * 设置暂停状态回调
     */
    public void setPauseStateCallback(PauseStateCallback callback) {
        this.pauseStateCallback = callback;
    }

    /**
     * [新增] 追加一个连接状态监听器（不会覆盖 setConnectionCallback 设置的主回调）
     */
    public void addConnectionListener(ConnectionCallback listener) {
        if (listener != null && !connectionListeners.contains(listener)) {
            connectionListeners.add(listener);
        }
    }

    /**
     * [新增] 移除连接状态监听器（页面 onDestroy 时务必调用，避免泄漏）
     */
    public void removeConnectionListener(ConnectionCallback listener) {
        connectionListeners.remove(listener);
    }

    /**
     * [新增] 追加一个暂停状态监听器（不会覆盖 setPauseStateCallback 设置的主回调）
     */
    public void addPauseStateListener(PauseStateCallback listener) {
        if (listener != null && !pauseStateListeners.contains(listener)) {
            pauseStateListeners.add(listener);
        }
    }

    /**
     * [新增] 移除暂停状态监听器
     */
    public void removePauseStateListener(PauseStateCallback listener) {
        pauseStateListeners.remove(listener);
    }

    // ====== [新增] 回调分发（统一切到主线程）======

    private void notifyConnectionState(boolean connected) {
        runOnMain(() -> {
            if (connectionCallback != null) connectionCallback.onConnectionStateChanged(connected);
            for (ConnectionCallback l : connectionListeners) l.onConnectionStateChanged(connected);
        });
    }

    private void notifyScanStarted() {
        runOnMain(() -> {
            if (connectionCallback != null) connectionCallback.onScanStarted();
            for (ConnectionCallback l : connectionListeners) l.onScanStarted();
        });
    }

    private void notifyScanFailed(String reason) {
        runOnMain(() -> {
            if (connectionCallback != null) connectionCallback.onScanFailed(reason);
            for (ConnectionCallback l : connectionListeners) l.onScanFailed(reason);
        });
    }

    private void notifyPauseState(boolean paused) {
        runOnMain(() -> {
            if (pauseStateCallback != null) pauseStateCallback.onPauseStateChanged(paused);
            for (PauseStateCallback l : pauseStateListeners) l.onPauseStateChanged(paused);
        });
    }

    private static void runOnMain(Runnable r) {
        new Handler(Looper.getMainLooper()).post(r);
    }

    /**
     * 检查蓝牙是否可用
     */
    public boolean isBluetoothAvailable() {
        return adapter != null && adapter.isEnabled();
    }

    /**
     * 检查是否已连接
     */
    public boolean isConnected() {
        return isConnected;
    }

    /**
     * 检查是否被暂停
     */
    public boolean isPausedByLocal() {
        return pausedByLocal;
    }

    /**
     * 开始扫描并连接
     */
    @SuppressLint("MissingPermission")
    public void startScanAndConnect() {
        if (scanner == null || adapter == null || !adapter.isEnabled()) {
            notifyScanFailed("蓝牙未开启或不可用");
            return;
        }

        // 检查权限
        if (!hasRequiredPermissions()) {
            notifyScanFailed("缺少蓝牙权限");
            return;
        }

        notifyScanStarted();

        Log.i(TAG, "开始扫描BLE设备...");

        List<ScanFilter> filters = new ArrayList<>();
        // 通过Service UUID过滤
        filters.add(new ScanFilter.Builder().setServiceUuid(new ParcelUuid(SERVICE_UUID)).build());

        ScanSettings settings = new ScanSettings.Builder()
                .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
                .build();

        scanner.startScan(filters, settings, scanCallback);

        // 设置扫描超时（20秒）
        scanTimeoutRunnable = () -> {
            stopScanSafe();
            if (!isConnected) {
                notifyScanFailed("扫描超时，未找到设备");
            }
        };
        scanTimeoutHandler.postDelayed(scanTimeoutRunnable, 20000);
    }

    /**
     * 扫描回调
     */
    private final ScanCallback scanCallback = new ScanCallback() {
        @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
        @Override
        public void onScanResult(int callbackType, ScanResult result) {
            BluetoothDevice dev = result.getDevice();
            String name = result.getScanRecord() != null ? result.getScanRecord().getDeviceName() : dev.getName();
            if (name != null && name.startsWith(TARGET_NAME_PREFIX)) {
                Log.i(TAG, "找到目标设备: " + name + " -> 连接中...");
                stopScanSafe();
                connectGatt(dev);
            }
        }

        @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
        @Override
        public void onBatchScanResults(List<ScanResult> results) {
            for (ScanResult r : results) onScanResult(0, r);
        }

        @Override
        public void onScanFailed(int errorCode) {
            Log.e(TAG, "扫描失败: " + errorCode);
            notifyScanFailed("扫描失败，错误码: " + errorCode);
        }
    };

    /**
     * 安全停止扫描
     */
    @SuppressLint("MissingPermission")
    private void stopScanSafe() {
        try {
            if (scanner != null) {
                scanner.stopScan(scanCallback);
            }
            if (scanTimeoutRunnable != null) {
                scanTimeoutHandler.removeCallbacks(scanTimeoutRunnable);
            }
        } catch (Exception e) {
            Log.e(TAG, "停止扫描异常", e);
        }
    }

    /**
     * 连接GATT
     */
    @SuppressLint("MissingPermission")
    private void connectGatt(BluetoothDevice device) {
        if (!hasRequiredPermissions()) return;
        gatt = device.connectGatt(context, false, gattCallback, BluetoothDevice.TRANSPORT_LE);
    }

    /**
     * GATT回调
     */
    private final BluetoothGattCallback gattCallback = new BluetoothGattCallback() {
        @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
        @Override
        public void onConnectionStateChange(BluetoothGatt g, int status, int newState) {
            Log.i(TAG, "连接状态变化: status=" + status + ", newState=" + newState);

            if (newState == BluetoothProfile.STATE_CONNECTED) {
                isConnected = true;
                Log.i(TAG, "已连接，协商MTU...");
                if (Build.VERSION.SDK_INT >= 21) {
                    g.requestMtu(247);
                } else {
                    g.discoverServices();
                }

                notifyConnectionState(true);
            } else if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                isConnected = false;
                isNotifying = false;
                setPaused(false);

                notifyConnectionState(false);
            }
        }

        @RequiresPermission(Manifest.permission.BLUETOOTH_CONNECT)
        @Override
        public void onMtuChanged(BluetoothGatt g, int mtu, int status) {
            Log.i(TAG, "MTU变更: mtu=" + mtu + ", status=" + status);
            g.discoverServices();
        }

        @Override
        public void onServicesDiscovered(BluetoothGatt g, int status) {
            Log.i(TAG, "服务发现: status=" + status);

            BluetoothGattService svc = g.getService(SERVICE_UUID);
            if (svc == null) {
                Log.e(TAG, "未找到指定服务");
                return;
            }

            rxChar = svc.getCharacteristic(RX_CHAR_UUID);
            txChar = svc.getCharacteristic(TX_CHAR_UUID);

            if (rxChar == null || txChar == null) {
                Log.e(TAG, "RX/TX特征缺失");
                return;
            }

            enableNotify(g, txChar);
        }

        @Override
        public void onCharacteristicChanged(BluetoothGatt g, BluetoothGattCharacteristic c) {
            if (TX_CHAR_UUID.equals(c.getUuid())) {
                byte[] data = c.getValue();
                parseIncomingFrame(data);
            }
        }

        @Override
        public void onCharacteristicWrite(BluetoothGatt g, BluetoothGattCharacteristic c, int status) {
            Log.d(TAG, "写入状态=" + status);
        }
    };

    /**
     * 启用通知
     */
    @SuppressLint("MissingPermission")
    private void enableNotify(BluetoothGatt g, BluetoothGattCharacteristic ch) {
        if (!hasRequiredPermissions()) return;

        boolean ok = g.setCharacteristicNotification(ch, true);
        BluetoothGattDescriptor cccd = ch.getDescriptor(CCCD_UUID);
        if (cccd != null) {
            cccd.setValue(BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE);
            g.writeDescriptor(cccd);
        }
        isNotifying = ok;
        Log.i(TAG, "订阅通知: " + ok);
    }

    /**
     * 发送动作命令
     * @param action "oral", "do", "Noise" 映射到 "000", "001", "002"
     */
    public void sendAction(String action, int finalFreq) {
        if (!isConnected || rxChar == null || gatt == null || pausedByLocal) {
            Log.w(TAG, "无法发送: connected=" + isConnected + ", paused=" + pausedByLocal);
            return;
        }

        // 将VideoProcessActivity的动作映射到蓝牙协议
        String mappedAction = mapAction(action);
        LEVEL = finalFreq;                                  // ★ 移到这里：先更新强度
        byte[] frame = buildFrameForAction(mappedAction);   // 再构建帧（读取新的 LEVEL）

        if (frame != null) {
            writeRx(frame);
            analysisRunning = !"002".equals(mappedAction); // "002" = 停止
            Log.i(TAG, "发送动作: " + action + " -> " + mappedAction
                    + ", pattern=" + analysisPattern + ", level=" + LEVEL);
        }
    }

    /**
     * 动作映射
     * 将VideoProcessActivity的动作映射到蓝牙测试app的动作
     */
    private String mapAction(String action) {
        // VideoProcessActivity输出: "oral", "do", "Noise"
        // 蓝牙测试app动作: "000", "001", "002"
        switch (action) {
            case "oral":
                return "000";    // oral -> 恒定-低
            case "do":
                return "001";    // doslow -> 脉冲-中
            case "Noise":
                return "002";    // Noise -> 停止
            default:
                return "002";    // 默认停止
        }
    }

    /**
     * 构建动作对应的帧
     */
    private byte[] buildFrameForAction(String action) {
        switch (action) {
            case "000": // 转（口交），模式由用户在手动电子菜单里选择
                return buildSetPatternFrame(analysisPattern, LEVEL, 0, 1);
            case "001": // 转（性爱），模式由用户在手动电子菜单里选择
                return buildSetPatternFrame(analysisPattern, LEVEL, 0, 1);
            //case "003": // 波形-中，循环
                //return buildSetPatternFrame(PATTERN_3, LEVEL, 0, 1);
            case "002": // 停止
                return buildStopAllFrame();
            default:
                return null;
        }
    }

    /**
     * [新增] 设置视频分析模式使用的变频模式（PATTERN_ID 1..3）。
     *
     * <p>视频分析链路只决定「转 / 不转 + 强度档位」，本方法决定「怎么转」。
     * 切换时只有当前正由视频分析驱动转动，才立即补发一帧让新模式生效；
     * 不转时只记下来，下一次发送时自然带上，避免在停机状态下把马达唤醒。</p>
     *
     * @param patternId 变频模式，1..3（协议 §9.1），超范围自动钳制
     */
    public void setAnalysisPattern(int patternId) {
        int p = clampPattern(patternId);
        if (p == analysisPattern) return;

        analysisPattern = p;
        Log.i(TAG, "视频分析变频模式更新为: " + p);

        boolean canResend = analysisRunning && LEVEL > 0
                && isConnected && !pausedByLocal && rxChar != null && gatt != null;
        if (canResend) {
            writeRx(buildSetPatternFrame(p, LEVEL, 0, 1));
            Log.i(TAG, "视频分析正在转动，已补发一帧: pattern=" + p + " level=" + LEVEL);
        }
    }

    /** [新增] 当前视频分析模式使用的变频模式（1..3） */
    public int getAnalysisPattern() {
        return analysisPattern;
    }

    /** [新增] 把变频模式钳制到协议允许的 1..3 */
    public static int clampPattern(int patternId) {
        if (patternId < PATTERN_MIN) return PATTERN_MIN;
        if (patternId > PATTERN_MAX) return PATTERN_MAX;
        return patternId;
    }

    /**
     * [新增] 手动电子菜单：直接下发「变频模式 + 马达强度」。
     * 等价于 buildSetPatternFrame(PATTERN, LEVEL, 0, 1)：
     * DURATION_MS=0 表示持续运行，FLAGS bit0=1 表示循环。
     *
     * @param patternId 变频模式，1..3（协议 §9.1）
     * @param intLevel  马达强度，0..10，0 为停止（协议 §9.2）
     * @return 帧已写入 RX 特征返回 true；未连接、被本地按键锁定或参数非法返回 false
     */
    public boolean sendManualPattern(int patternId, int intLevel) {
        if (!isConnected || rxChar == null || gatt == null) {
            Log.w(TAG, "手动控制发送失败: 蓝牙未连接");
            return false;
        }
        if (pausedByLocal) {
            // 锁定期内设备会对 SetPattern 回 BUSY，这里直接拦下，由 UI 提示先「恢复控制」
            Log.w(TAG, "手动控制发送失败: 设备处于本地按键锁定");
            return false;
        }
        if (patternId < PATTERN_MIN || patternId > PATTERN_MAX) {
            Log.w(TAG, "手动控制发送失败: 非法模式 " + patternId);
            return false;
        }
        int level = Math.max(LEVEL_MIN, Math.min(LEVEL_MAX, intLevel));

        writeRx(buildSetPatternFrame(patternId, level, 0, 1));
        analysisRunning = false; // 手动接管，后续切模式不再自动补发
        Log.i(TAG, "手动控制: pattern=" + patternId + " level=" + level);
        return true;
    }

    /**
     * [新增] 手动电子菜单：紧急停机。
     * StopAll 优先级最高，锁定期内设备也必须接受，因此这里不检查 pausedByLocal。
     *
     * @return 帧已写入 RX 特征返回 true
     */
    public boolean sendStopAll() {
        if (!isConnected || rxChar == null || gatt == null) {
            Log.w(TAG, "停止命令发送失败: 蓝牙未连接");
            return false;
        }
        writeRx(buildStopAllFrame());
        analysisRunning = false;
        Log.i(TAG, "手动控制: StopAll");
        return true;
    }

    /**
     * 发送恢复控制命令
     */
    public void sendResumeControl() {
        if (!isConnected || rxChar == null || gatt == null) {
            Log.w(TAG, "无法发送恢复命令: 未连接");
            return;
        }

        byte[] frame = buildResumeFrame();
        writeRx(frame);
        Log.i(TAG, "发送恢复控制命令");
    }

    /**
     * 写入数据到RX特征
     */
    @SuppressLint("MissingPermission")
    private void writeRx(byte[] frame) {
        if (rxChar == null || gatt == null || !hasRequiredPermissions()) return;

        rxChar.setWriteType(BluetoothGattCharacteristic.WRITE_TYPE_NO_RESPONSE);
        rxChar.setValue(frame);
        gatt.writeCharacteristic(rxChar);
    }

    /**
     * 断开连接
     */
    @SuppressLint("MissingPermission")
    public void disconnect() {
        stopScanSafe();

        if (gatt != null) {
            if (hasRequiredPermissions()) {
                gatt.disconnect();
                gatt.close();
            }
            gatt = null;
        }

        isConnected = false;
        isNotifying = false;
        analysisRunning = false;
        setPaused(false);

        notifyConnectionState(false);
    }

    /**
     * 清理资源
     */
    public void close() {
        disconnect();
        connectionCallback = null;
        pauseStateCallback = null;
        connectionListeners.clear();
        pauseStateListeners.clear();
    }

    // ====== 帧编码（协议实现）======

    private byte[] buildSetPatternFrame(int patternId, int intLevel, int durationMs, int flags) {
        ByteBuffer payload = ByteBuffer.allocate(5).order(ByteOrder.LITTLE_ENDIAN);
        payload.put((byte) patternId);
        payload.put((byte) intLevel);
        payload.putShort((short) durationMs);
        payload.put((byte) (flags & 0xFF));
        return buildFrame(CMD_SET_PATTERN, payload.array());
    }

    private byte[] buildStopAllFrame() {
        return buildFrame(CMD_STOP_ALL, new byte[0]);
    }

    private byte[] buildResumeFrame() {
        return buildFrame(CMD_RESUME_APP, new byte[0]);
    }

    private byte[] buildFrame(int cmd, byte[] payload) {
        int len = payload.length;
        ByteBuffer bb = ByteBuffer.allocate(2 + 1 + 1 + 1 + 2 + len + 2).order(ByteOrder.LITTLE_ENDIAN);

        bb.put((byte) 0xAA).put((byte) 0x55);        // SOF
        bb.put((byte) VER);                          // VER
        bb.put((byte) (cmd & 0xFF));                 // CMD
        bb.put((byte) (seq & 0xFF));                 // SEQ
        bb.putShort((short) len);                    // LEN
        bb.put(payload);                             // PAYLOAD

        // 计算CRC16
        byte[] crcInput = new byte[1 + 1 + 1 + 2 + len];
        ByteBuffer tmp = ByteBuffer.wrap(crcInput).order(ByteOrder.LITTLE_ENDIAN);
        tmp.put((byte) VER).put((byte) (cmd & 0xFF)).put((byte) (seq & 0xFF));
        tmp.putShort((short) len).put(payload);
        int crc = crc16Ccitt(tmp.array());
        bb.putShort((short) crc);

        seq++;
        return bb.array();
    }

    /**
     * CRC16-CCITT计算
     */
    private static int crc16Ccitt(byte[] data) {
        int crc = 0xFFFF;
        for (byte b : data) {
            crc ^= ((b & 0xFF) << 8);
            for (int i = 0; i < 8; i++) {
                if ((crc & 0x8000) != 0) {
                    crc = (crc << 1) ^ 0x1021;
                } else {
                    crc <<= 1;
                }
                crc &= 0xFFFF;
            }
        }
        return crc & 0xFFFF;
    }

    /**
     * 解析设备上行帧
     */
    private void parseIncomingFrame(byte[] pkt) {
        try {
            if (pkt == null || pkt.length < 2 + 1 + 1 + 1 + 2 + 2) return;

            ByteBuffer bb = ByteBuffer.wrap(pkt).order(ByteOrder.LITTLE_ENDIAN);
            byte sof0 = bb.get(), sof1 = bb.get();
            if (sof0 != (byte) 0xAA || sof1 != (byte) 0x55) return;

            int ver = bb.get() & 0xFF;
            int cmd = bb.get() & 0xFF;
            int rseq = bb.get() & 0xFF;
            int len = bb.getShort() & 0xFFFF;

            if (len < 0 || len > 512 || pkt.length < 2 + 1 + 1 + 1 + 2 + len + 2) return;

            byte[] payload = new byte[len];
            bb.get(payload);
            int crcLe = bb.getShort() & 0xFFFF;

            // 校验CRC
            ByteBuffer tmp = ByteBuffer.allocate(1 + 1 + 1 + 2 + len).order(ByteOrder.LITTLE_ENDIAN);
            tmp.put((byte) ver).put((byte) cmd).put((byte) rseq);
            tmp.putShort((short) len).put(payload);
            int crc = crc16Ccitt(tmp.array());

            if (crc != crcLe) {
                Log.w(TAG, "CRC校验失败");
                return;
            }

            // 处理ACK
            if (cmd == (0x80 + CMD_SET_PATTERN) || cmd == (0x80 + CMD_STOP_ALL) ||
                    cmd == (0x80 + CMD_RESUME_APP)) {

                if (payload.length >= 2) {
                    int ackSeq = payload[0] & 0xFF;
                    int status = payload[1] & 0xFF;
                    Log.i(TAG, String.format(Locale.US, "ACK cmd=0x%02X seq=%d status=%d", cmd, ackSeq, status));

                    // 锁定期：ACK=BUSY -> 进入暂停态
                    if (status == 1) { // BUSY
                        setPaused(true);
                    }

                    // 恢复命令成功
                    if ((cmd == (0x80 + CMD_RESUME_APP)) && status == 0) {
                        setPaused(false);
                    }
                }
            } else if (cmd == CMD_STATE_RPT) {
                // StateReport处理
                parseStateReport(payload);
                Log.i(TAG, "收到StateReport len=" + payload.length);
            } else {
                Log.i(TAG, String.format(Locale.US, "收到命令 cmd=0x%02X len=%d", cmd, len));
            }
        } catch (Exception e) {
            Log.e(TAG, "解析帧错误", e);
        }
    }

    /**
     * 解析StateReport扩展字段
     */
    private void parseStateReport(byte[] p) {
        try {
            int off = 0;
            if (p.length < 12) return;

            off += 2; // FW_VER
            off += 2; // BAT_mV
            off += 2; // TEMP_dC
            off += 1; // CH_CNT
            off += 1; // CUR_PATTERN
            off += 1; // CUR_INTLVL
            off += 2; // RUN_REMAIN
            off += 1; // FLAGS

            // 扩展字段
            if (p.length >= off + 9) {
                int rev = ((p[off] & 0xFF) | ((p[off+1] & 0xFF) << 8));
                off += 2;
                int src = (p[off++] & 0xFF);
                int chgMask = (p[off++] & 0xFF);
                int owner = (p[off++] & 0xFF);
                int holdMode = (p[off++] & 0xFF);
                int holdTtl = ((p[off] & 0xFF) | ((p[off+1] & 0xFF) << 8));
                off += 2;
                int btnCode = (p[off++] & 0xFF);

                Log.i(TAG, "StateReport: src=" + src + " owner=" + owner +
                        " holdMode=" + holdMode + " holdTtl=" + holdTtl);

                // 本地按键触发手动恢复模式
                if (src == SRC_BUTTON && holdMode == HOLD_MANUAL) {
                    setPaused(true);
                }

                // 解除锁定
                if (holdMode == HOLD_NONE) {
                    setPaused(false);
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "解析StateReport错误", e);
        }
    }

    /**
     * 设置暂停状态
     */
    private void setPaused(boolean paused) {
        pausedByLocal = paused;
        notifyPauseState(paused);
        Log.i(TAG, "暂停状态变更: " + paused);
    }

    /**
     * 检查是否有必要的权限
     */
    private boolean hasRequiredPermissions() {
        if (Build.VERSION.SDK_INT >= 31) {
            return ContextCompat.checkSelfPermission(context,
                    Manifest.permission.BLUETOOTH_SCAN) == PackageManager.PERMISSION_GRANTED &&
                    ContextCompat.checkSelfPermission(context,
                            Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED;
        } else {
            return ContextCompat.checkSelfPermission(context,
                    Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED;
        }
    }
}