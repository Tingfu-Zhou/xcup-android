package com.example.helloworld;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

/**
 * LevelBoost（定时加档）的单元测试：计时只算在跑的时间、播完 / 退出清零、加档不唤醒停着的马达并钳到最高档。
 */
public class LevelBoostTest {

    private static final long MIN = 60_000L;
    private static final long T = 5_000_000L;   // 任意起点（单调时钟）

    @Test
    public void boostsOnceDelayReached() {
        LevelBoost b = new LevelBoost(true, 10, 2);
        b.resume(T);
        assertEquals(0, b.boostAt(T));
        assertEquals(0, b.boostAt(T + 10 * MIN - 1));
        assertEquals(2, b.boostAt(T + 10 * MIN));
        assertEquals(2, b.boostAt(T + 90 * MIN));    // 到点后一直加，直到 reset
    }

    @Test
    public void disabledNeverBoosts() {
        LevelBoost b = new LevelBoost(false, 1, 2);
        b.resume(T);
        assertEquals(0, b.boostAt(T + 120 * MIN));
    }

    @Test
    public void pausedTimeDoesNotCount() {
        LevelBoost b = new LevelBoost(true, 10, 1);
        b.resume(T);
        b.pause(T + 6 * MIN);                        // 播了 6 分钟后暂停
        assertEquals(0, b.boostAt(T + 30 * MIN));    // 暂停期间不计时
        b.resume(T + 30 * MIN);
        assertEquals(0, b.boostAt(T + 34 * MIN - 1));
        assertEquals(1, b.boostAt(T + 34 * MIN));    // 6 + 4 = 10 分钟
    }

    @Test
    public void repeatedResumeAndPauseAreIdempotent() {
        // 融合循环每个 tick 都调 resume（在跑）或 pause（暂停中）：重复调用不能重新起表，也不能重复累计
        LevelBoost b = new LevelBoost(true, 1, 1);
        b.resume(T);
        b.resume(T + 30_000);
        assertEquals(MIN, b.elapsedMs(T + MIN));
        b.pause(T + MIN);
        b.pause(T + 2 * MIN);
        assertEquals(MIN, b.elapsedMs(T + 5 * MIN));
    }

    @Test
    public void resetStartsOver() {
        LevelBoost b = new LevelBoost(true, 1, 2);
        b.resume(T);
        assertEquals(2, b.boostAt(T + MIN));
        b.reset();                                   // 视频播完 / 退出模式
        assertEquals(0, b.boostAt(T + 2 * MIN));     // 清零后停表，等下一次 resume
        b.resume(T + 3 * MIN);
        assertEquals(0, b.boostAt(T + 4 * MIN - 1));
        assertEquals(2, b.boostAt(T + 4 * MIN));
    }

    @Test
    public void applyKeepsStopAndClampsToMax() {
        assertEquals(0, LevelBoost.apply(0, 2, 10));     // 0 档（停）不加
        assertEquals(5, LevelBoost.apply(5, 0, 10));
        assertEquals(6, LevelBoost.apply(5, 1, 10));
        assertEquals(10, LevelBoost.apply(8, 2, 10));    // 分析最高 8 档，+2 正好到 10
        assertEquals(10, LevelBoost.apply(9, 2, 10));    // 超过最高档的钳到最高档
    }

    @Test
    public void settingsAreClamped() {
        LevelBoost b = new LevelBoost(true, 0, 5);       // 存储里的坏值
        b.resume(T);
        assertEquals(0, b.boostAt(T + MIN - 1));         // 最短 1 分钟
        assertEquals(2, b.boostAt(T + MIN));             // 最多 +2
        assertEquals("BOOST=+2@1min", b.paramsLine());
        assertEquals("BOOST=off", new LevelBoost(false, 10, 1).paramsLine());
        assertEquals(LevelBoost.MAX_DELAY_MIN, LevelBoost.clampDelayMin(999));
        assertEquals(LevelBoost.MIN_LEVELS, LevelBoost.clampLevels(-1));
    }
}
