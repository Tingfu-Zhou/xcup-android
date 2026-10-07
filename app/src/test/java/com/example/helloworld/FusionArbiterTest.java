package com.example.helloworld;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

/**
 * FusionArbiter 的单元测试。
 *
 * <p>最重要的是 {@link #matchesPythonGolden()}：bench/tools/fusion_lab.py 的 FusionArbiter 是 Java 版的逐行移植，
 * 离线回放、扫参、check 都靠它；两边逐 tick 不一致，离线评估出的效果在手机上就不成立。对拍数据由
 * {@code python3 bench/tools/fusion_lab.py golden} 生成（真实 run 的若干段 + 合成边界用例）。</p>
 */
public class FusionArbiterTest {

    @Test
    public void matchesPythonGolden() throws IOException {
        InputStream in = getClass().getClassLoader().getResourceAsStream("fusion/arbiter_golden.csv");
        assertNotNull("缺少测试资源 fusion/arbiter_golden.csv（python3 bench/tools/fusion_lab.py golden 生成）", in);
        BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
        assertEquals("reset,now,v_action,v_conf,v_ts,a_action,a_conf,a_ts,motor_on,exp_action,exp_reason",
                r.readLine());
        FusionArbiter arb = new FusionArbiter();
        String line;
        int n = 0;
        while ((line = r.readLine()) != null) {
            if (line.isEmpty()) continue;
            String[] f = line.split(",", -1);
            String row = "golden 第 " + (n + 2) + " 行";
            if (!f[0].isEmpty()) {                       // reset 列：<ms> 或 <ms>+<预热ms>
                int plus = f[0].indexOf('+');
                if (plus < 0) {
                    arb.reset(Long.parseLong(f[0]));
                } else {
                    arb.reset(Long.parseLong(f[0].substring(0, plus)), Long.parseLong(f[0].substring(plus + 1)));
                }
            }
            String got = arb.decide(Long.parseLong(f[1]),
                    f[2], Float.parseFloat(f[3]), Long.parseLong(f[4]),
                    f[5], Float.parseFloat(f[6]), Long.parseLong(f[7]),
                    "1".equals(f[8]));
            assertEquals(row + " action", f[9], got);
            assertEquals(row + " reason", f[10], arb.lastReason());
            n++;
        }
        r.close();
        assertTrue("对拍数据太少: " + n, n >= 5000);
    }

    private static final long T = 1_700_000_000_000L;

    /** 维持时主动交 "do"：有支持就不会交 ""，蓝牙状态机里没执行的停转计时会被取消。 */
    @Test
    public void holdReassertsDo() {
        FusionArbiter arb = new FusionArbiter();
        arb.reset(T);
        // 音频 4 个结果里 3 个 "转"：有支持，交 do
        for (int i = 0; i < 3; i++) {
            arb.decide(T + 1000 * (i + 1), "Noise", 0.99f, T + 1000 * (i + 1) - 10,
                    "do", 0.8f, T + 1000 * (i + 1) - 5, true);
        }
        String a = arb.decide(T + 4000, "Noise", 0.99f, T + 3990, "Noise", 0.9f, T + 3995, true);
        assertEquals("do", a);
        assertEquals(FusionArbiter.R_HOLD_AUDIO, arb.lastReason());
        // 再来一个 "不转"：4 个里只剩 2 个 "转"，没有支持且有结束证据 -> Noise
        String b = arb.decide(T + 5000, "Noise", 0.99f, T + 4990, "Noise", 0.9f, T + 4995, true);
        assertEquals("Noise", b);
        assertEquals(FusionArbiter.R_STOP_NEG, arb.lastReason());
    }

    /** 同一个结果在多个 tick 里出现只算一次：一个音频结果撑不起"4 个里 3 个"。 */
    @Test
    public void duplicatesCountOnce() {
        FusionArbiter arb = new FusionArbiter();
        arb.reset(T);
        for (int i = 0; i < 6; i++) {
            arb.decide(T + 500 * i, "", 0f, -1, "do", 0.95f, T + 100, true);
        }
        String a = arb.decide(T + 3000, "", 0f, -1, "do", 0.95f, T + 100, true);
        // 只有 1 个音频结果：不构成支持；这一路结果还不够判断 -> 继续 do（有上限）
        assertEquals("do", a);
        assertEquals(FusionArbiter.R_WAIT_SPARSE, arb.lastReason());
    }

    /** 刚恢复播放（reset 后马达仍在转）时正好是特写：音频攒够结果之前，视频的"剧情"不能把马达停掉。 */
    @Test
    public void resumeInCloseUpDoesNotStop() {
        FusionArbiter arb = new FusionArbiter();
        arb.reset(T);
        for (int k = 0; k < 16; k++) {
            long now = T + 250 + 500L * k;
            long aTs = k >= 2 ? T + 1000L * (k / 2) - 100 : -1;
            String g = arb.decide(now, "Noise", 0.9f, now - 100, aTs > 0 ? "do" : "", 0.95f, aTs, true);
            assertEquals("tick " + k + " " + arb.lastReason(), "do", g);
        }
        assertEquals(FusionArbiter.R_HOLD_AUDIO, arb.lastReason());
    }

    /** 两路都是"转""不转"交替的强证据：不满足维持条件就不启动，避免每 2 秒开关一次。 */
    @Test
    public void alternatingStrongEvidenceDoesNotStart() {
        FusionArbiter arb = new FusionArbiter();
        arb.reset(T);
        for (int k = 0; k < 40; k++) {
            long now = T + 500L * (k + 1);
            String v = k % 2 == 0 ? "do" : "Noise";
            String a = (k / 2) % 2 == 0 ? "do" : "Noise";
            String g = arb.decide(now, v, 0.95f, now - 50, a, 0.95f, now - 100 - 500L * (k % 2), false);
            assertTrue("tick " + k + " " + arb.lastReason(), !"do".equals(g));
        }
    }

    /** 墙钟往回跳：自动清空证据；"未来"的旧结果不记，也不挡住之后的新结果。 */
    @Test
    public void clockJumpBackResets() {
        FusionArbiter arb = new FusionArbiter();
        arb.reset(T);
        for (int k = 1; k <= 4; k++) {
            arb.decide(T + 1000L * k, "", 0f, -1, "do", 0.95f, T + 1000L * k - 10, true);
        }
        assertEquals(FusionArbiter.R_HOLD_AUDIO, arb.lastReason());
        long back = T - 60000;
        arb.decide(back, "", 0f, -1, "do", 0.95f, T + 3990, true);            // 跳回 60 秒：旧结果在"未来"
        assertEquals(FusionArbiter.R_WAIT_SPARSE, arb.lastReason());
        String g = "";
        for (int k = 1; k <= 4; k++) {
            g = arb.decide(back + 1000L * k, "Noise", 0.9f, back + 1000L * k - 20,
                    "Noise", 0.9f, back + 1000L * k - 10, true);
        }
        assertEquals("Noise", g);
        assertEquals(FusionArbiter.R_STOP_NEG, arb.lastReason());
    }

    /** 看不清 / 听不清 / 过期时最多再维持 UNCLEAR_MAX_MS，然后停。 */
    @Test
    public void unclearHoldIsFinite() {
        FusionArbiter arb = new FusionArbiter();
        arb.reset(T);
        assertEquals("do", arb.decide(T, "", 0f, -1, "", 0f, -1, true));
        assertEquals("do", arb.decide(T + FusionArbiter.UNCLEAR_MAX_MS - 1, "", 0f, -1, "", 0f, -1, true));
        assertEquals("Noise", arb.decide(T + FusionArbiter.UNCLEAR_MAX_MS, "", 0f, -1, "", 0f, -1, true));
        assertEquals(FusionArbiter.R_STOP_UNCLEAR, arb.lastReason());
    }

    /** reset(now) 之后，now 之前产出的结果（seek 前的旧结果）不再算证据。 */
    @Test
    public void resetDropsOlderResults() {
        FusionArbiter arb = new FusionArbiter();
        arb.reset(T);
        String a = arb.decide(T + 100, "do", 0.99f, T - 50, "do", 0.99f, T - 20, false);
        assertEquals("", a);
        assertEquals(FusionArbiter.R_OFF_WAIT, arb.lastReason());
        String b = arb.decide(T + 600, "do", 0.99f, T + 500, "do", 0.99f, T + 550, false);
        assertEquals("", b);                                   // 视频强"转"只有 1 个，还不够
        String c = arb.decide(T + 1100, "do", 0.99f, T + 1000, "do", 0.99f, T + 550, false);
        assertEquals("do", c);
        assertEquals(FusionArbiter.R_START_BOTH, arb.lastReason());
    }

    /** 音频兜底零容忍：9 秒内夹一个 unclear 就不启动。 */
    @Test
    public void audioValveNeedsUninterruptedPositives() {
        FusionArbiter arb = new FusionArbiter();
        arb.reset(T);
        String last = "";
        for (int i = 0; i < 4; i++) {
            last = arb.decide(T + 1000 * (i + 1), "Noise", 0.99f, T + 1000 * (i + 1) - 1,
                    i == 1 ? "" : "do", 0.7f, T + 1000 * (i + 1), false);
        }
        assertEquals("Noise", last);
        for (int i = 4; i < 14; i++) {
            last = arb.decide(T + 1000 * (i + 1), "Noise", 0.99f, T + 1000 * (i + 1) - 1,
                    "do", 0.7f, T + 1000 * (i + 1), false);
            if ("do".equals(last)) break;
        }
        assertEquals("do", last);
        assertEquals(FusionArbiter.R_START_AUDIO, arb.lastReason());
    }

    @Test
    public void paramsLineCarriesVersion() {
        String p = FusionArbiter.paramsLine();
        assertTrue(p, p.startsWith("ARB=v1 "));
        assertTrue(p, p.contains("ARB_Z_WIN=9000"));
        assertTrue(p, p.contains("ARB_V_FLOOR=0.8500"));
        assertTrue(p, p.contains("ARB_START_V_CONS=80"));
    }
}
