package com.example.helloworld;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.junit.Test;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

/**
 * EvidenceFusion 的单元测试。
 *
 * <p>最重要的是 {@link #matchesPythonGolden()}：Java 端与 bench/tools/fusion_lab.py 的 Python 实现
 * 必须逐 tick 一致，否则离线回放扫出来的参数在手机上就不是那个效果了。对拍数据由
 * {@code python3 bench/tools/fusion_lab.py golden} 生成。</p>
 */
public class EvidenceFusionTest {

    private static String resource(String name) throws IOException {
        InputStream in = EvidenceFusionTest.class.getClassLoader().getResourceAsStream(name);
        assertNotNull("缺少测试资源 " + name + "（python3 bench/tools/fusion_lab.py golden 生成）", in);
        ByteArrayOutputStream buf = new ByteArrayOutputStream();
        byte[] b = new byte[8192];
        int n;
        while ((n = in.read(b)) > 0) buf.write(b, 0, n);
        in.close();
        return new String(buf.toByteArray(), StandardCharsets.UTF_8);
    }

    private static float[] probs(String s) {
        if (s.isEmpty()) return null;
        if (s.equals("-")) return new float[0];
        String[] p = s.split("\\|");
        float[] r = new float[p.length];
        for (int i = 0; i < p.length; i++) r[i] = Float.parseFloat(p[i]);
        return r;
    }

    private static void assertLlr(String row, String want, double got) {
        if (want.isEmpty()) {
            assertTrue(row + " 期望未使用证据，实际 " + got, Double.isNaN(got));
        } else {
            assertEquals(row, Double.parseDouble(want), got, 1e-12);
        }
    }

    @Test
    public void matchesPythonGolden() throws Exception {
        EvidenceFusion.Config cfg = EvidenceFusion.Config.fromJson(resource("fusion/golden_config.json"));
        EvidenceFusion ef = new EvidenceFusion(cfg);
        BufferedReader r = new BufferedReader(new StringReader(resource("fusion/golden.csv")));
        assertEquals("reset,now,v_ts,v_probs,a_ts,a_probs,exp_action,exp_l,exp_llr_v,exp_llr_a",
                r.readLine());
        String line;
        int n = 0;
        while ((line = r.readLine()) != null) {
            if (line.isEmpty()) continue;
            String[] f = line.split(",", -1);
            String row = "golden 第 " + (n + 2) + " 行";
            if ("1".equals(f[0])) ef.reset();
            long now = Long.parseLong(f[1]);
            EvidenceFusion.Snapshot v = f[2].isEmpty() ? null
                    : new EvidenceFusion.Snapshot(probs(f[3]), Long.parseLong(f[2]));
            EvidenceFusion.Snapshot a = f[4].isEmpty() ? null
                    : new EvidenceFusion.Snapshot(probs(f[5]), Long.parseLong(f[4]));
            EvidenceFusion.Output out = ef.update(v, a, now);
            assertEquals(row + " action", f[6], out.action);
            assertEquals(row + " L", Double.parseDouble(f[7]), out.l, 1e-12);
            assertLlr(row + " llrV", f[8], out.llrV);
            assertLlr(row + " llrA", f[9], out.llrA);
            n++;
        }
        assertTrue("对拍数据太少: " + n, n >= 400);
    }

    private static EvidenceFusion.Config simple() {
        return new EvidenceFusion.Config(
                new EvidenceFusion.Modality(new double[]{0, 1, 1}, new double[]{0, 0.5, 1},
                        new double[]{-1.0, 1.0}),
                new EvidenceFusion.Modality(new double[]{1, 1, 0}, new double[]{0, 0.5, 1},
                        new double[]{-0.5, 0.5}),
                1.0, 0.5, 2.0, 2.0, 1.5, -0.5, 2000, EvidenceFusion.MODE_SHADOW, "t", false);
    }

    private static final float[] V_DO = {0.1f, 0.2f, 0.7f};
    private static final float[] V_NO = {0.9f, 0.05f, 0.05f};

    private static EvidenceFusion.Snapshot v(float[] p, long ts) {
        return new EvidenceFusion.Snapshot(p, ts);
    }

    @Test
    public void hysteresisDedupeAndClamp() {
        EvidenceFusion ef = new EvidenceFusion(simple());
        assertEquals("", ef.update(null, null, 1000).action);
        assertEquals("Noise", ef.update(v(V_DO, 900), null, 1000).action);       // L=1 < 1.5
        assertEquals(1.0, ef.update(v(V_DO, 900), null, 1500).l, 0);              // 同一结果不重复计
        assertEquals("do", ef.update(v(V_DO, 1400), null, 1500).action);         // L=2 ≥ 1.5
        assertEquals(2.0, ef.update(v(V_DO, 1900), null, 2000).l, 0);             // clamp
        assertEquals("do", ef.update(v(V_NO, 2400), null, 2500).action);         // L=1 迟滞
        assertEquals("do", ef.update(v(V_NO, 2900), null, 3000).action);         // L=0 迟滞
        assertEquals("Noise", ef.update(v(V_NO, 3400), null, 3500).action);      // L=-1 ≤ -0.5
    }

    /** 恰好等于阈值时也要切换（>= / <=），与 Python 端一致。 */
    @Test
    public void thresholdsAreInclusive() {
        EvidenceFusion ef = new EvidenceFusion(simple());
        float[] aDo = {0.7f, 0.2f, 0.1f};
        float[] aNo = {0.05f, 0.05f, 0.9f};
        ef.update(v(V_DO, 900), v(aDo, 900), 1000);                                // 1 + 0.25
        EvidenceFusion.Output o = ef.update(null, v(aDo, 1400), 1500);             // + 0.25 = 1.5
        assertEquals(1.5, o.l, 0);
        assertEquals("do", o.action);
        ef.update(v(V_NO, 1900), null, 2000);                                      // 0.5
        ef.update(v(V_NO, 2400), null, 2500);                                      // -0.5 = thetaOff
        o = ef.update(null, null, 3000);
        assertEquals(-0.5, o.l, 0);
        assertEquals("Noise", o.action);
        ef.update(null, v(aNo, 2900), 3000);                                       // -0.75，保持不转
        assertEquals("Noise", ef.update(null, null, 3500).action);
    }

    @Test
    public void staleInvalidAndOldResultsGiveNoEvidence() {
        EvidenceFusion ef = new EvidenceFusion(simple());
        ef.update(v(V_DO, 3400), null, 3500);
        assertTrue(Double.isNaN(ef.update(v(V_DO, 100), null, 4000).llrV));            // 比已用的旧
        assertTrue(Double.isNaN(ef.update(v(V_DO, 3500), null, 6000).llrV));           // 过期 2500ms
        assertTrue(Double.isNaN(ef.update(v(new float[3], 5900), null, 6000).llrV));   // 全 0
        assertTrue(Double.isNaN(ef.update(v(null, 5950), null, 6000).llrV));           // null
        assertTrue(Double.isNaN(ef.update(null, v(new float[0], 5950), 6000).llrA));   // 空数组
        assertEquals(-0.5, ef.update(null, v(new float[]{0.05f, 0.05f, 0.9f}, 5960), 6000).llrA, 0);
    }

    @Test
    public void resetReturnsToNoDecision() {
        EvidenceFusion ef = new EvidenceFusion(simple());
        ef.update(v(V_DO, 900), null, 1000);
        ef.update(v(V_DO, 1400), null, 1500);
        ef.reset();
        EvidenceFusion.Output o = ef.update(null, null, 2000);
        assertEquals("", o.action);
        assertEquals(0.0, o.l, 0);
    }

    @Test
    public void rejectsInvalidConfig() {
        EvidenceFusion.Modality m = new EvidenceFusion.Modality(new double[]{0, 1, 1},
                new double[]{0, 1}, new double[]{0});
        try {
            new EvidenceFusion.Config(m, m, 1, 1, 2, 2, 0.5, 1.0, 2000, "shadow", "x", false);
            fail("theta_off >= theta_on 应报错");
        } catch (IllegalArgumentException expected) {
            // ok
        }
        try {
            new EvidenceFusion.Config(m, m, 1, 1, 2, 2, 1.0, -1.0, 2000, "on", "x", false);
            fail("非法 mode 应报错");
        } catch (IllegalArgumentException expected) {
            // ok
        }
        try {
            new EvidenceFusion.Config(m, m, 1, 1, 1, 4, 1.0, -1.0, 2000, "shadow", "x", false);
            fail("theta_off <= -clamp_lo 应报错");
        } catch (IllegalArgumentException expected) {
            // ok
        }
        try {
            new EvidenceFusion.Modality(new double[]{1}, new double[]{0, 0.5, 1}, new double[]{0});
            fail("edges/llr 长度不匹配应报错");
        } catch (IllegalArgumentException expected) {
            // ok
        }
    }

    /** 上下界分开：剧情里 L 最低到 -clampLo，动作里最高到 +clampHi。 */
    @Test
    public void asymmetricClamp() {
        EvidenceFusion.Config c = simple();
        EvidenceFusion ef = new EvidenceFusion(new EvidenceFusion.Config(c.video, c.audio, 1.0, 0.0,
                1.0, 3.0, 0.5, -0.5, 2000, EvidenceFusion.MODE_SHADOW, "t", false));
        EvidenceFusion.Output o = null;
        for (int i = 0; i < 6; i++) o = ef.update(v(V_NO, 100 + i), null, 200 + i);
        assertEquals(-1.0, o.l, 0);
        for (int i = 0; i < 6; i++) o = ef.update(v(V_DO, 300 + i), null, 400 + i);
        assertEquals(3.0, o.l, 0);
    }

    /** 只有对称 clamp 的旧配置仍能解析（上下界都取 clamp）。 */
    @Test
    public void legacySymmetricClampKey() throws Exception {
        String json = resource("fusion/golden_config.json")
                .replaceAll("\"clamp_lo\":\\s*[0-9.]+,", "")
                .replaceAll("\"clamp_hi\":\\s*[0-9.]+,", "\"clamp\": 2.5,");
        EvidenceFusion.Config cfg = EvidenceFusion.Config.fromJson(json);
        assertEquals(2.5, cfg.clampLo, 0);
        assertEquals(2.5, cfg.clampHi, 0);
    }

    /** 打进 APK 的配置必须能解析，否则 app 会静默回退到原融合。 */
    @Test
    public void shippedConfigParses() throws Exception {
        File f = new File("src/main/assets/fusion_config.json");          // Gradle 单测工作目录 = app/
        if (!f.exists()) f = new File("app/src/main/assets/fusion_config.json");
        assertTrue("找不到 " + f.getAbsolutePath(), f.exists());
        String text = new String(Files.readAllBytes(f.toPath()), StandardCharsets.UTF_8);
        EvidenceFusion.Config cfg = EvidenceFusion.Config.fromJson(text);
        assertNotNull(cfg.id);
    }
}
