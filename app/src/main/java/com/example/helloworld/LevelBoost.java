package com.example.helloworld;

/**
 * 定时加档（模拟临近高潮）：进入离线视频 / 在线录屏模式后，分析累计跑满用户选定的时间，
 * 视频分析给出的档位自动再加 1 或 2 档。在手动电子菜单里开关，默认关闭（设置的读写见 ManualControlActivity）。
 *
 * <p><b>计时</b>只累计融合循环在跑的时间：视频暂停、静音暂停分析时 {@link #pause}，恢复后 {@link #resume} 接着计；
 * seek 不影响。退出当前模式、视频播放完毕时 {@link #reset} 从头计时。在线模式看不到视频什么时候播完，
 * 连续静音到清空分析缓冲（约 10 秒）就当播完。</p>
 *
 * <p><b>加档</b>只加在发给设备的档位上：蓝牙状态机（updateBluetoothState）的 currentLevel 仍是分析档位，迟滞、档位确认都不变，
 * 发送时才用 {@link #apply} 加上；lastSentLevel 记的是发出去的档位，所以到点那一刻档位没变也会按状态机原有的节流
 * （距上次发送 ≥ 1600ms）补发一次，不会另起一路发送。0 档（停）仍是 0，不会把停着的马达唤醒；其余加上后最高到 10 档。
 * 分析链路自己最高只给到 8 档，所以 9、10 档只有加档之后才会用到。没打开时发出去的档位与原来完全相同，离线回放
 * （fusion_lab.py check）不受影响；打开后录的 run 到点后 sent_level 是加档后的档位，check 的 sent 列会对不上。</p>
 *
 * <p>纯 Java、不依赖 Android（时钟由调用方传入），单元测试 LevelBoostTest。每个分析页面持有一个实例，只在融合线程（主线程）调用。</p>
 */
public final class LevelBoost {

    /** 开始加档的时间（分钟）。范围与 activity_manual_control.xml 里 sliderBoostDelay 的 valueFrom / valueTo 一致 */
    public static final int MIN_DELAY_MIN = 1;
    public static final int MAX_DELAY_MIN = 30;
    public static final int DEFAULT_DELAY_MIN = 10;
    /** 加几档 */
    public static final int MIN_LEVELS = 1;
    public static final int MAX_LEVELS = 2;
    public static final int DEFAULT_LEVELS = 1;

    private final boolean enabled;
    private final int delayMin;
    private final int levels;

    private long accumulatedMs = 0;     // 之前各段（已暂停）累计的时长
    private long runningSinceMs = -1;   // 这一段开始计时的时刻；-1 = 没在计时

    public LevelBoost(boolean enabled, int delayMin, int levels) {
        this.enabled = enabled;
        this.delayMin = clampDelayMin(delayMin);
        this.levels = clampLevels(levels);
    }

    /** 融合循环在跑：开始或接着计时。已经在计时就什么都不做（每个 tick 都会调） */
    public void resume(long nowMs) {
        if (runningSinceMs < 0) runningSinceMs = nowMs;
    }

    /** 分析暂停：停表，已经计的时间保留。没在计时就什么都不做 */
    public void pause(long nowMs) {
        if (runningSinceMs < 0) return;
        accumulatedMs += Math.max(0L, nowMs - runningSinceMs);
        runningSinceMs = -1;
    }

    /** 退出模式 / 视频播完：清零并停表，下一次 resume 从头计时 */
    public void reset() {
        accumulatedMs = 0;
        runningSinceMs = -1;
    }

    /** 累计计了多久（毫秒） */
    public long elapsedMs(long nowMs) {
        long running = runningSinceMs < 0 ? 0L : Math.max(0L, nowMs - runningSinceMs);
        return accumulatedMs + running;
    }

    /** 此刻该加几档：没打开或还没到点为 0 */
    public int boostAt(long nowMs) {
        return enabled && elapsedMs(nowMs) >= delayMin * 60_000L ? levels : 0;
    }

    /** 评测 CSV 参数行里的一项：BOOST=off，或 BOOST=+&lt;档数&gt;@&lt;分钟&gt;min */
    public String paramsLine() {
        return enabled ? "BOOST=+" + levels + "@" + delayMin + "min" : "BOOST=off";
    }

    /**
     * 把加档加到分析档位上：0 档（停）不加，其余加上后钳到 maxLevel。
     *
     * @param level    分析链路给出的档位
     * @param boost    {@link #boostAt} 的结果
     * @param maxLevel 协议允许的最高档（BLEManager.LEVEL_MAX）
     */
    public static int apply(int level, int boost, int maxLevel) {
        if (level <= 0 || boost <= 0) return level;
        return Math.min(maxLevel, level + boost);
    }

    /** 存储里的旧值 / 坏值钳到 MIN_DELAY_MIN..MAX_DELAY_MIN（Slider 设超范围的值会抛异常） */
    public static int clampDelayMin(int delayMin) {
        return Math.max(MIN_DELAY_MIN, Math.min(MAX_DELAY_MIN, delayMin));
    }

    /** 钳到 MIN_LEVELS..MAX_LEVELS */
    public static int clampLevels(int levels) {
        return Math.max(MIN_LEVELS, Math.min(MAX_LEVELS, levels));
    }
}
