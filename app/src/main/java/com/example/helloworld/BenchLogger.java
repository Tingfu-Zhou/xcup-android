package com.example.helloworld;

import android.content.Context;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageInfo;
import android.net.Uri;
import android.util.Log;
import android.widget.Toast;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * 评测模式数据采集器。
 *
 * <p>把每个融合 tick 的 raw probs / 决策中间结果 / 蓝牙动作写成一行 CSV，交给
 * bench/tools/metrics.py 与人工真值对齐，算出逐秒准确率与切换延迟。
 *
 * <p><b>设计约束：只读不写，且不扰动被测对象。</b>本类不参与任何判定；融合循环跑在主线程，
 * 所以 tick() 只做字符串拼接后投进队列，文件 I/O 全在后台线程，避免落盘抖动污染我们正要
 * 测量的那个延迟。队列满时丢行并计数，绝不阻塞主线程。关闭时所有方法都是空操作。
 *
 * <p><b>时间轴用视频播放位置 pos_ms</b>，而不是墙钟。真值也是视频位置，两者天然对齐，
 * 不需要任何标定。墙钟 wall_ms 另存一列，因为蓝牙状态机的计时用的是墙钟。
 *
 * <p>输出路径：{@code /sdcard/Android/data/<包名>/files/bench/run_<视频名>_<时间>.csv}，
 * 用数据线或文件管理器直接拷走，不需要任何存储权限。
 *
 * <p>重要：采集时请<b>从头不间断播到尾，不要 seek、不要暂停</b>。seek 会清空滑窗与
 * 音频缓冲并重置状态机，使这一轮的延迟数字失去意义（metrics.py 会检测 pos 回退并告警）。
 */
public final class BenchLogger {

    private static final String TAG = "BenchLogger";

    /** release 包想抓数据时改成 true。debug 包（可调试包）自动开启。 */
    private static final boolean FORCE_ENABLE = false;

    private static final BenchLogger INSTANCE = new BenchLogger();

    public static BenchLogger get() {
        return INSTANCE;
    }

    private BenchLogger() {}

    private boolean enabled = false;
    private File dir;
    private String videoName = "run";
    private String paramsLine = "";
    private BufferedWriter out;          // 只由后台写入线程访问（open() 里创建后才启动该线程）
    private volatile boolean failed = false;
    private Context appCtx;
    private int rowCount = 0;            // 已投递行数（仅主线程访问）
    private int dropped = 0;             // 队列满而丢弃的行数

    /** 容量按 ~17 分钟的 tick 量给（500ms/行），正常情况下永远不会满。 */
    private final BlockingQueue<String> queue = new LinkedBlockingQueue<>(2048);
    private static final String POISON = "\u0000EOF";
    private Thread writerThread;

    // 本 tick 内蓝牙是否发了指令（updateBluetoothState 里置位，tick() 写完后清零）。
    // noteSend 与 tick 都只在融合线程（主线程）调用，无需同步。
    private boolean sent = false;
    private String sentAction = "";
    private int sentLevel = -1;

    /** 在 onCreate 里调一次。enabled=false 时后面所有调用都是空操作。 */
    public void init(Context ctx, Uri videoUri, String paramsLine) {
        this.enabled = FORCE_ENABLE || isDebuggable(ctx);
        if (!enabled) return;
        // 本类是单例，二次进入视频页时必须清掉上一轮的残留。
        this.sent = false;
        this.sentAction = "";
        this.sentLevel = -1;
        this.rowCount = 0;
        this.dropped = 0;
        this.failed = false;
        this.appCtx = ctx.getApplicationContext();
        this.paramsLine = paramsLine == null ? "" : paramsLine;
        this.videoName = sanitize(guessName(videoUri));
        File base = ctx.getExternalFilesDir(null);
        this.dir = base == null ? null : new File(base, "bench");
        Log.i(TAG, "评测模式已开启，视频=" + videoName
                + "，输出目录=" + (dir == null ? "(不可用)" : dir.getAbsolutePath()));
    }

    public boolean isEnabled() {
        return enabled;
    }

    /** 在实际决定下发指令处调用（不要放在「BLE 已连接」分支里，否则不插杯子就采不到数据）。 */
    public void noteSend(String action, int level) {
        if (!enabled) return;
        sent = true;
        sentAction = action == null ? "" : action;
        sentLevel = level;
    }

    /**
     * 每个融合 tick 末尾调一次，一行一个 tick。
     *
     * <p>vProbs/vAgeMs（及音频的）必须来自同一个模型结果快照：离线回放靠 wall_ms - age 还原
     * "这组概率是哪一刻出的"来去重，概率和年龄错配会让回放与真机对不上。</p>
     *
     * @param ev 证据融合本 tick 的输出；未启用时传 null（ev_* 列留空）
     */
    public void tick(long posMs, long wallMs,
                     String vAction, float vConf, long vAgeMs, float[] vProbs,
                     String aAction, float aConf, long aAgeMs, float[] aProbs,
                     int loudLevel, float loudConf, boolean loudValid, long loudAgeMs,
                     String smoothed, String gated, int finalFreq,
                     String bleState, int bleLevel,
                     EvidenceFusion.Output ev) {
        if (!enabled || failed) return;
        if (out == null && !open()) return;

        float[] vp = vProbs, ap = aProbs;
        StringBuilder sb = new StringBuilder(260);
        sb.append(posMs).append(',').append(wallMs).append(',');
        sb.append(csv(vAction)).append(',').append(f(vConf)).append(',').append(age(vAgeMs)).append(',');
        sb.append(p(vp, 0)).append(',').append(p(vp, 1)).append(',').append(p(vp, 2)).append(',');
        sb.append(csv(aAction)).append(',').append(f(aConf)).append(',').append(age(aAgeMs)).append(',');
        sb.append(p(ap, 0)).append(',').append(p(ap, 1)).append(',').append(p(ap, 2)).append(',');
        sb.append(loudLevel).append(',').append(f(loudConf)).append(',')
          .append(loudValid ? 1 : 0).append(',').append(age(loudAgeMs)).append(',');
        sb.append(csv(smoothed)).append(',').append(csv(gated)).append(',').append(finalFreq).append(',');
        sb.append(csv(bleState)).append(',').append(bleLevel).append(',')
          .append(sent ? 1 : 0).append(',').append(csv(sentAction)).append(',').append(sentLevel);
        if (ev != null) {
            sb.append(',').append(csv(ev.action)).append(',').append(d(ev.l)).append(',')
              .append(d(ev.llrV)).append(',').append(d(ev.llrA));
        } else {
            sb.append(",,,,");
        }

        if (queue.offer(sb.toString())) {
            rowCount++;
        } else {
            dropped++;                               // 宁可丢一行，也不在主线程上等 I/O
            if (dropped == 1) Log.w(TAG, "队列已满，开始丢行（不影响已写入的数据）");
        }
        sent = false;
        sentAction = "";
        sentLevel = -1;
    }

    /** 在 onDestroy 里调用：排空队列、落盘、提示文件路径。 */
    public void stop() {
        if (!enabled || writerThread == null) return;
        File f = currentFile;
        int n = rowCount, d = dropped;
        queue.offer(POISON);
        try {
            writerThread.join(3000);                 // 等后台把剩下的行写完
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
        }
        writerThread = null;
        String msg = "评测数据已保存：" + (f == null ? "?" : f.getAbsolutePath())
                + "，共 " + n + " 行" + (d > 0 ? "（丢弃 " + d + " 行）" : "");
        Log.i(TAG, msg);
        toast("评测数据已保存 (" + n + " 行" + (d > 0 ? "，丢 " + d : "") + ")\n"
                + (f == null ? "" : f.getAbsolutePath()));
    }

    /** 后台写入线程：唯一接触 out 的地方。 */
    private void drain() {
        int written = 0;
        try {
            while (true) {
                String line = queue.take();
                if (POISON.equals(line)) break;
                out.write(line);
                out.write('\n');
                if (++written % 40 == 0) out.flush();   // 约每 20 秒落盘，中途崩了也不丢太多
            }
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();
        } catch (Exception e) {
            Log.e(TAG, "写入失败，停止采集", e);
            failed = true;
        }
        close();
    }

    // ------------------------------------------------------------------ 内部
    private File currentFile;

    private boolean open() {
        if (dir == null) {
            Log.e(TAG, "外部文件目录不可用，评测采集关闭");
            failed = true;
            return false;
        }
        try {
            if (!dir.exists() && !dir.mkdirs()) {
                Log.e(TAG, "创建目录失败: " + dir);
                failed = true;
                return false;
            }
            String ts = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
            currentFile = new File(dir, "run_" + videoName + "_" + ts + ".csv");
            out = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(currentFile), "UTF-8"));
            out.write("# xcup-bench-run v1\n");
            out.write("# video_name=" + videoName + "\n");
            out.write("# app_version=" + appVersion(appCtx) + "\n");
            out.write("# device=" + android.os.Build.MANUFACTURER + "_" + android.os.Build.MODEL
                    + " sdk=" + android.os.Build.VERSION.SDK_INT + "\n");
            out.write("# started_at=" + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
                    .format(new Date()) + "\n");
            out.write("# params: " + paramsLine + "\n");
            out.write("pos_ms,wall_ms,"
                    + "v_action,v_conf,v_age,v_p_plot,v_p_oral,v_p_sex,"
                    + "a_action,a_conf,a_age,a_p_sex,a_p_oral,a_p_noise,"
                    + "loud_lv,loud_conf,loud_valid,loud_age,"
                    + "smoothed,gated,final_freq,"
                    + "ble_state,ble_level,sent,sent_action,sent_level,"
                    + "ev_action,ev_l,ev_llr_v,ev_llr_a\n");
            out.flush();
            writerThread = new Thread(this::drain, "BenchLoggerWriter");
            writerThread.setDaemon(true);
            writerThread.start();
            Log.i(TAG, "评测数据写入: " + currentFile.getAbsolutePath());
            toast("评测模式：正在采集\n" + currentFile.getName());
            return true;
        } catch (Exception e) {
            Log.e(TAG, "打开输出文件失败，评测采集关闭", e);
            failed = true;
            out = null;
            return false;
        }
    }

    private void close() {
        if (out != null) {
            try { out.flush(); out.close(); } catch (Exception ignored) {}
            out = null;
        }
    }

    private void toast(final String msg) {
        if (appCtx == null) return;
        try {
            new android.os.Handler(android.os.Looper.getMainLooper())
                    .post(() -> Toast.makeText(appCtx, msg, Toast.LENGTH_LONG).show());
        } catch (Exception ignored) {}
    }

    /** 不依赖生成类 BuildConfig，避免受 buildFeatures.buildConfig 开关影响。 */
    private static boolean isDebuggable(Context ctx) {
        try {
            ApplicationInfo ai = ctx.getApplicationInfo();
            return ai != null && (ai.flags & ApplicationInfo.FLAG_DEBUGGABLE) != 0;
        } catch (Exception e) {
            return false;
        }
    }

    private static String appVersion(Context ctx) {
        if (ctx == null) return "?";
        try {
            PackageInfo pi = ctx.getPackageManager().getPackageInfo(ctx.getPackageName(), 0);
            return pi == null || pi.versionName == null ? "?" : pi.versionName;
        } catch (Exception e) {
            return "?";
        }
    }

    private static String guessName(Uri uri) {
        if (uri == null) return "run";
        String s = uri.getLastPathSegment();
        if (s == null) s = uri.toString();
        int slash = s.lastIndexOf('/');
        if (slash >= 0) s = s.substring(slash + 1);
        int dot = s.lastIndexOf('.');
        if (dot > 0) s = s.substring(0, dot);
        return s.isEmpty() ? "run" : s;
    }

    /** 文件名里只留安全字符；中文片名也常见，所以保留非 ASCII 字母。 */
    private static String sanitize(String s) {
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < s.length() && b.length() < 48; i++) {
            char c = s.charAt(i);
            if (Character.isLetterOrDigit(c) || c == '-' || c == '_') b.append(c);
            else b.append('_');
        }
        return b.length() == 0 ? "run" : b.toString();
    }

    private static String csv(String s) {
        if (s == null || s.isEmpty()) return "";
        return s.indexOf(',') >= 0 ? s.replace(',', ';') : s;
    }

    private static String f(float v) {
        return Float.isNaN(v) ? "" : String.format(Locale.US, "%.4f", v);
    }

    private static String d(double v) {
        return Double.isNaN(v) ? "" : String.format(Locale.US, "%.4f", v);
    }

    /** 过期时长可能是 Long.MAX_VALUE（从未出结果），统一写 -1。 */
    private static String age(long ms) {
        return (ms < 0 || ms >= Long.MAX_VALUE / 2) ? "-1" : Long.toString(ms);
    }

    private static String p(float[] probs, int i) {
        return (probs == null || i >= probs.length) ? "" : f(probs[i]);
    }
}
