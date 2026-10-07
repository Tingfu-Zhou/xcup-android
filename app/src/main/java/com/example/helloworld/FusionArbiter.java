package com.example.helloworld;

import java.util.Locale;

/**
 * 融合仲裁器：把"启动 / 维持 / 停止"分开判断，决定每个融合 tick 交给蓝牙状态机什么动作。
 * 2026-10-07 起替代原来的 applyDoStartGate，离线 / 网页视频模式与在线模式共用。
 *
 * <p><b>为什么改</b>（bench 7 部片）：原融合把"音频更可信"写死（投票权重 1.2 / 0.8，音频 ≥0.75 直接启动）。
 * 动作中 22 次误停全部是"只有一个模态说剧情"：视频遇到特写 / 遮挡判剧情时，音频一个 1 秒的 noise 就让马达停；
 * 或者视频稳稳判动作、音频听不到目标声音，马达也停。剧情里的误转则主要来自音频单独判"转"（约 43%）。</p>
 *
 * <p><b>证据</b>一律按"新推理结果"计数：同一个结果在多个 500ms tick 里重复出现只算一次（按结果时间戳去重），
 * 产出时已超过 2 秒的结果不记。维持 / 停止按"最近 n 个结果"计数，手机推理慢一些也不会失效。</p>
 *
 * <p><b>启动</b>（马达不转），满足任一条才交 "do"：</p>
 * <ol>
 *   <li>两路佐证：最近 {@link #C_WIN_MS} 内，视频强"转"结果（置信度 ≥ {@link #V_FLOOR}）≥ {@link #C_V} 个，
 *       且音频强"转"结果（≥ {@link #A_FLOOR}）≥ {@link #C_A} 个，并且此刻满足比"维持"更严一档的条件
 *       （视频"转"占比 ≥ {@link #START_V_CONS_PCT}%，或音频最近 {@link #MA_N} 个里 ≥ {@link #START_MA_K} 个"转"）。
 *       启动比维持严（迟滞），防止"一启动就停"：证据在维持门槛上下晃时，不加它马达会每几秒开关一次。</li>
 *   <li>音频兜底：最近 {@link #Z_WIN_MS} 内的音频结果至少 {@link #Z_N} 个，而且<b>全部</b>判"转"
 *       （夹一个"不转"或 unclear 就不算）。这条不看视频，因为特写 / 遮挡时视频判剧情，不能因此永远不启动；
 *       而剧情里的音频常是"转"里每隔几秒夹一个"不转"，零容忍正好把它挡住。</li>
 * </ol>
 * <p>都不满足时：最近 {@link #NEG_WIN_MS} 内有"不转"结果交 "Noise"，否则交 ""（不驱动状态机）。</p>
 *
 * <p><b>维持</b>（马达在转），任一模态"有支持"就交 "do"：</p>
 * <ul>
 *   <li>音频支持：最近 {@link #MA_N} 个音频结果（都在 {@link #M_MAX_MS} 内）里"转" ≥ {@link #MA_K} 个。
 *       可以容忍一个 1 秒的 noise，但一个孤立的"转"撑不住。</li>
 *   <li>视频支持：最近 {@link #MV_N} 个视频结果（都在 {@link #M_MAX_MS} 内）里"转" ≥ {@link #MV_K} 个，
 *       并且最近 {@link #CONS_WIN_MS} 内视频结果里"转"的占比 ≥ {@link #V_CONS_PCT}%。这条是一致性检查：
 *       交替出现的零星"转"不能无限续持。</li>
 * </ul>
 * <p>维持时<b>主动交 "do"</b>而不是 ""。这样蓝牙状态机里尚未执行的停转计时会被取消，之后的 Noise 必须重新稳定 1 秒
 * 才会停；如果交 ""，旧计时会保留下来，下一次 Noise 立刻停转。</p>
 *
 * <p><b>停止</b>（马达在转、两路都没有支持）：</p>
 * <ul>
 *   <li>reset（seek / 恢复播放）之后的 {@link #COLD_MS} 内，某一路的新结果还不够 MA_N / MV_N 个，说明它还没法判断，
 *       先继续交 "do" 等新结果。否则恢复播放时恰好在特写，视频一个"剧情"就会在音频攒够结果之前把马达停掉。
 *       只在刚 reset 时这样等：平时某一路变慢或断掉，不能让另一路时有时无的支持把马达一直留着。</li>
 *   <li>两路都判断得了：最近 {@link #NEG_WIN_MS} 内有任一"不转"结果，交 "Noise"
 *       （之后照旧要状态机稳定 1 秒 + 最短驻留 2 秒）。</li>
 *   <li>看不清 / 听不清 / 结果过期时继续交 "do"。上面两种等待都从最后一次有支持算起，最多 {@link #UNCLEAR_MAX_MS}，
 *       然后交 "Noise"。原融合在两路都过期时会一直转下去。</li>
 * </ul>
 *
 * <p>墙钟往回跳（用户改时间 / 网络校时）时，账本里的时间戳都会变成"未来"，这里会自动 reset。
 * 注意蓝牙状态机（updateBluetoothState）的计时也用墙钟，往回跳一大步时它会卡住相应的时长，那是原有的问题，这里不处理。</p>
 *
 * <p>纯 Java、不依赖 Android。bench/tools/fusion_lab.py 里的 {@code FusionArbiter} 是它的逐行移植，
 * 离线回放、扫参、check 都靠它；FusionArbiterTest 用 {@code python3 bench/tools/fusion_lab.py golden}
 * 生成的对拍数据校验两边逐 tick 一致。改这里的规则或参数时，两边要一起改，并重新生成对拍数据。</p>
 *
 * <p>线程：只在融合线程（主线程）调用，不加锁。</p>
 */
public final class FusionArbiter {

    // ======================================================== 参数（与 fusion_lab.py 的 ARB_DEFAULTS 一致）
    /** 结果产出后超过这么久就不再记账（与融合循环的新鲜度窗 MAX_AGE 相同）。 */
    static final long MAX_AGE_MS = 2000;
    /** 两路佐证启动时，视频 / 音频"强转"结果的置信度下限。 */
    static final float V_FLOOR = 0.85f;
    static final float A_FLOOR = 0.90f;
    /** 启动·两路佐证：窗口、需要的视频强"转"个数、需要的音频强"转"个数。 */
    static final long C_WIN_MS = 3000;
    static final int C_V = 2;
    static final int C_A = 1;
    /** 启动·音频兜底：Z_WIN_MS 内的音频结果至少 Z_N 个，且全部判"转"。 */
    static final long Z_WIN_MS = 9000;
    static final int Z_N = 4;
    /** 维持：音频看最近 MA_N 个结果、视频看最近 MV_N 个结果，都须在 M_MAX_MS 内。 */
    static final int MA_N = 4;
    static final int MA_K = 3;
    static final int MV_N = 4;
    static final int MV_K = 2;
    static final long M_MAX_MS = 12000;   // 给结果来得慢的手机留余量：音频 3 秒一个时，最近 4 个仍落在窗口内
    /** 维持·视频一致性：CONS_WIN_MS 内视频结果里"转"的占比下限（百分数，用整数运算，避免两端浮点边界不一致）。 */
    static final long CONS_WIN_MS = 6000;
    static final int V_CONS_PCT = 70;
    /** 停止：多久内出现过"不转"结果，算有结束证据。 */
    static final long NEG_WIN_MS = 4000;
    /** 停止：既没有支持、也没有结束证据（看不清 / 听不清 / 过期）时，最多再维持多久。 */
    static final long UNCLEAR_MAX_MS = 6000;
    /** 停止：reset 之后多久内，结果还不够判断的一路可以让马达先等（只管刚 seek / 恢复播放的冷启动）。 */
    static final long COLD_MS = 6000;
    /** 启动·两路佐证还要满足的"更严一档"的维持条件（迟滞）：视频"转"占比、音频最近 MA_N 个里"转"的个数。 */
    static final int START_V_CONS_PCT = 80;
    static final int START_MA_K = 4;
    /** 静音暂停后恢复时，丢掉恢复后这么久内产出的结果（模型窗口里还混着暂停前的静音）。音频窗口 2 秒。 */
    public static final long SILENCE_RESUME_WARMUP_MS = 2000;
    /** 账本只需保留最长窗口（Z_WIN_MS / M_MAX_MS）以内的结果，这里留足余量。 */
    static final long KEEP_MS = 20000;

    // ======================================================== 本 tick 输出的原因（评测 CSV 的 arb 列）
    public static final String R_START_BOTH = "start_both";
    public static final String R_START_AUDIO = "start_audio";
    public static final String R_OFF_NEG = "off_neg";
    public static final String R_OFF_WAIT = "off_wait";
    public static final String R_HOLD_BOTH = "hold_both";
    public static final String R_HOLD_VIDEO = "hold_video";
    public static final String R_HOLD_AUDIO = "hold_audio";
    public static final String R_STOP_NEG = "stop_neg";
    public static final String R_STOP_UNCLEAR = "stop_unclear";
    public static final String R_WAIT_UNCLEAR = "wait_unclear";
    public static final String R_WAIT_SPARSE = "wait_sparse";

    /** 动作串 -> 类别：1 = "转"(do)，-1 = "不转"(Noise)，0 = unclear。 */
    static int cls(String action) {
        if ("do".equals(action)) return 1;
        if ("Noise".equals(action)) return -1;
        return 0;
    }

    /**
     * 一次模型推理的判定结果。分析线程把它作为<b>一个</b>对象原子发布，融合线程只读这一个对象：
     * 动作串、置信度和产出时刻必须来自同一次推理，否则按时间戳去重时会把旧动作记在新结果名下、
     * 再把真正的新结果当成重复丢掉。
     */
    public static final class Result {
        public final String action;   // "do" / "Noise" / ""（unclear）
        public final float conf;
        public final long tsMs;       // 产出时刻（墙钟）

        public Result(String action, float conf, long tsMs) {
            this.action = action == null ? "" : action;
            this.conf = conf;
            this.tsMs = tsMs;
        }
    }

    /** 一个模态的"新结果"账本：环形缓冲，按结果时间戳去重。 */
    static final class Ledger {
        private static final int CAP = 128;      // 视频最多约 2 个/秒 × KEEP_MS 20 秒，留足余量
        private final long[] ts = new long[CAP];
        private final int[] cls = new int[CAP];
        private final float[] conf = new float[CAP];
        private int head = 0;                    // 最旧一条的下标
        private int size = 0;
        private long lastTs = Long.MIN_VALUE;
        private long minTs = Long.MIN_VALUE;

        /** 清空；产出时刻早于 minTs 的结果（seek / 重置之前的旧结果）以后也不再记。 */
        void reset(long minTs) {
            head = 0;
            size = 0;
            lastTs = Long.MIN_VALUE;
            this.minTs = minTs;
        }

        /**
         * 每个 tick 喂一次当前最新结果；resultTs <= 0 表示还没有结果。返回是否记了一条新结果。
         * 产出时刻晚于 now 的结果（只会在墙钟往回跳之后出现）不记、也不推进去重游标，免得挡住之后的新结果。
         */
        boolean feed(String action, float c, long resultTs, long now) {
            if (resultTs <= 0 || resultTs > now || resultTs <= lastTs) return false;
            lastTs = resultTs;
            if (resultTs < minTs || now - resultTs > MAX_AGE_MS) return false;
            if (size == CAP) {                   // 正常到不了；万一满了丢最旧的
                head = (head + 1) % CAP;
                size--;
            }
            int i = (head + size) % CAP;
            ts[i] = resultTs;
            cls[i] = cls(action);
            conf[i] = c;
            size++;
            long lim = now - KEEP_MS;
            while (size > 0 && ts[head] <= lim) {
                head = (head + 1) % CAP;
                size--;
            }
            return true;
        }

        /** 第 k 新的一条（k = 0 为最新）。 */
        private int idx(int k) {
            return (head + size - 1 - k) % CAP;
        }

        /** (now - win, now] 内类别为 c 且置信度 ≥ floor 的结果数。 */
        int count(long now, long win, int c, float floor) {
            long lo = now - win;
            int n = 0;
            for (int k = 0; k < size; k++) {
                int i = idx(k);
                if (ts[i] <= lo) break;
                if (cls[i] == c && conf[i] >= floor) n++;
            }
            return n;
        }

        /** (now - win, now] 内的结果总数。 */
        int total(long now, long win) {
            long lo = now - win;
            int n = 0;
            for (int k = 0; k < size; k++) {
                if (ts[idx(k)] <= lo) break;
                n++;
            }
            return n;
        }

        /** 最近 n 个结果里落在 (now - maxWin, now] 内的有几个（≤ n）。 */
        int recent(long now, int n, long maxWin) {
            long lo = now - maxWin;
            int got = 0;
            for (int k = 0; k < size && k < n; k++) {
                if (ts[idx(k)] <= lo) break;
                got++;
            }
            return got;
        }

        /** 最近 n 个结果（须在 (now - maxWin, now] 内）里判"转"的个数。 */
        int lastNPos(long now, int n, long maxWin) {
            long lo = now - maxWin;
            int pos = 0;
            for (int k = 0; k < size && k < n; k++) {
                int i = idx(k);
                if (ts[i] <= lo) break;
                if (cls[i] == 1) pos++;
            }
            return pos;
        }
    }

    private final Ledger video = new Ledger();
    private final Ledger audio = new Ledger();
    private long lastSupportMs = -1;     // 维持阶段最后一次有支持的时刻；马达不转时为 -1
    private long prevNowMs = Long.MIN_VALUE;
    private long resetAtMs = Long.MIN_VALUE;   // 上次 reset 的时刻（冷启动窗口从这里算）
    private String reason = "";

    /** seek、暂停后恢复时调用：清空证据，nowMs 之前产出的结果以后都不算。 */
    public void reset(long nowMs) {
        reset(nowMs, 0);
    }

    /**
     * 同上，另外把 nowMs + warmupMs 之前产出的结果也丢掉。静音暂停后恢复时用：那时模型窗口里还有暂停前的静音，
     * 头一两个结果不可信（音频兜底启动要求窗口内全是"转"，一个陈旧的"不转"就能把启动拖后 9 秒）。
     */
    public void reset(long nowMs, long warmupMs) {
        video.reset(nowMs + warmupMs);
        audio.reset(nowMs + warmupMs);
        lastSupportMs = -1;
        prevNowMs = Long.MIN_VALUE;
        resetAtMs = nowMs;
        reason = "";
    }

    /** 本 tick 输出的原因（写进评测 CSV 的 arb 列）。 */
    public String lastReason() {
        return reason;
    }

    /**
     * 上一个 tick 交 "do" 是不是有证据撑着（启动 / 维持），而不是在"等结果"（wait_sparse / wait_unclear）。
     * 暂停时用它决定马达要不要带过暂停：只是在等结果的 "do" 不能带过去。否则网页视频模式下
     * "静音暂停 → 一声短响恢复（恢复挂起的 do）→ 还在等结果时又静音暂停（又挂起 do）"会让马达跟着每一声短响一直转。
     */
    public boolean lastDoHadEvidence() {
        return reason.startsWith("start_") || reason.startsWith("hold_");
    }

    /**
     * 每个融合 tick 调一次。
     *
     * @param now     本 tick 的墙钟（与评测 CSV 的 wall_ms 相同）
     * @param vAction 视频动作串（"do"/"Noise"/""，已按新鲜度过滤）
     * @param vConf   视频置信度
     * @param vTs     该视频结果的产出时刻（结果时间戳），还没有结果传 -1
     * @param motorOn 蓝牙状态机当前是否在转（currentBluetoothState 是 do）
     * @return "do" / "Noise" / ""（"" = 本 tick 不驱动蓝牙状态机）
     */
    public String decide(long now, String vAction, float vConf, long vTs,
                         String aAction, float aConf, long aTs, boolean motorOn) {
        if (prevNowMs != Long.MIN_VALUE && now < prevNowMs) reset(now);   // 墙钟往回跳
        prevNowMs = now;
        video.feed(vAction, vConf, vTs, now);
        audio.feed(aAction, aConf, aTs, now);

        if (!motorOn) {
            lastSupportMs = -1;
            if (video.count(now, C_WIN_MS, 1, V_FLOOR) >= C_V
                    && audio.count(now, C_WIN_MS, 1, A_FLOOR) >= C_A
                    && (videoSupport(now, START_V_CONS_PCT)
                        || audio.lastNPos(now, MA_N, M_MAX_MS) >= START_MA_K)) {
                reason = R_START_BOTH;
                return "do";
            }
            int n = audio.total(now, Z_WIN_MS);
            if (n >= Z_N && audio.count(now, Z_WIN_MS, 1, 0f) == n) {
                reason = R_START_AUDIO;
                return "do";
            }
            if (hasNeg(now)) {
                reason = R_OFF_NEG;
                return "Noise";
            }
            reason = R_OFF_WAIT;
            return "";
        }

        if (lastSupportMs < 0) lastSupportMs = now;
        boolean sa = audio.lastNPos(now, MA_N, M_MAX_MS) >= MA_K;
        boolean sv = videoSupport(now, V_CONS_PCT);
        if (sv || sa) {
            lastSupportMs = now;
            reason = sv && sa ? R_HOLD_BOTH : (sv ? R_HOLD_VIDEO : R_HOLD_AUDIO);
            return "do";
        }
        boolean waiting = now - lastSupportMs < UNCLEAR_MAX_MS;
        boolean cold = resetAtMs != Long.MIN_VALUE && now - resetAtMs < COLD_MS;
        if (waiting && cold
                && (audio.recent(now, MA_N, M_MAX_MS) < MA_N || video.recent(now, MV_N, M_MAX_MS) < MV_N)) {
            reason = R_WAIT_SPARSE;
            return "do";
        }
        if (hasNeg(now)) {
            reason = R_STOP_NEG;
            return "Noise";
        }
        if (!waiting) {
            reason = R_STOP_UNCLEAR;
            return "Noise";
        }
        reason = R_WAIT_UNCLEAR;
        return "do";
    }

    /** 视频支持：最近 MV_N 个结果里"转" ≥ MV_K 个，且 CONS_WIN_MS 内"转"占比 ≥ consPct%。 */
    private boolean videoSupport(long now, int consPct) {
        if (video.lastNPos(now, MV_N, M_MAX_MS) < MV_K) return false;
        int n = video.total(now, CONS_WIN_MS);
        int p = video.count(now, CONS_WIN_MS, 1, 0f);
        return n > 0 && p * 100 >= consPct * n;
    }

    private boolean hasNeg(long now) {
        return video.count(now, NEG_WIN_MS, -1, 0f) + audio.count(now, NEG_WIN_MS, -1, 0f) > 0;
    }

    /** 写进评测 CSV 的参数行。fusion_lab.py 靠 ARB= 判断这份 run 用的是哪套仲裁、参数是什么。 */
    public static String paramsLine() {
        return String.format(Locale.US,
                "ARB=v1 ARB_V_FLOOR=%.4f ARB_A_FLOOR=%.4f ARB_C_WIN=%d ARB_C_V=%d ARB_C_A=%d "
                        + "ARB_Z_WIN=%d ARB_Z_N=%d ARB_MA_N=%d ARB_MA_K=%d "
                        + "ARB_MV_N=%d ARB_MV_K=%d ARB_M_MAX=%d ARB_CONS_WIN=%d ARB_V_CONS=%d "
                        + "ARB_NEG_WIN=%d ARB_UNCLEAR_MAX=%d ARB_MAX_AGE=%d ARB_KEEP=%d "
                        + "ARB_COLD=%d ARB_START_V_CONS=%d ARB_START_MA_K=%d",
                V_FLOOR, A_FLOOR, C_WIN_MS, C_V, C_A, Z_WIN_MS, Z_N, MA_N, MA_K,
                MV_N, MV_K, M_MAX_MS, CONS_WIN_MS, V_CONS_PCT, NEG_WIN_MS, UNCLEAR_MAX_MS,
                MAX_AGE_MS, KEEP_MS, COLD_MS, START_V_CONS_PCT, START_MA_K);
    }
}
