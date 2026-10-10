package com.example.helloworld;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

/**
 * AudioCue 的单元测试。
 *
 * <p>{@link #matchesPythonGolden()}：bench/tools/fusion_lab.py 的 cue_features / audio_cue 是逐行移植，离线回放和
 * 选阈值都靠它。对拍数据由 {@code python3 bench/tools/fusion_lab.py golden} 生成：真实结果（YAMNet 分数）、
 * 阈值的单精度值和前后 1 ulp、各下标数量级悬殊的随机值。分组分数逐位比较。</p>
 */
public class AudioCueTest {

    @Test
    public void matchesPythonGolden() throws IOException {
        InputStream in = getClass().getClassLoader().getResourceAsStream("fusion/audio_cue_golden.csv");
        assertNotNull("缺少测试资源 fusion/audio_cue_golden.csv（python3 bench/tools/fusion_lab.py golden 生成）", in);
        BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
        String[] head = r.readLine().split(",", -1);
        // 表头：s<下标>… 是线索用到的那几个 AudioSet 下标，然后是分组分数列（与 BenchLogger 的列名相同）和线索名
        int nIdx = head.length - AudioCue.N_FEATURES - 1;
        int[] idx = new int[nIdx];
        for (int j = 0; j < nIdx; j++) {
            assertTrue(head[j], head[j].startsWith("s"));
            idx[j] = Integer.parseInt(head[j].substring(1));
        }
        assertArrayEquals(AudioCue.FEATURE_COLS, Arrays.copyOfRange(head, nIdx, nIdx + AudioCue.N_FEATURES));
        assertEquals("a_cue", head[head.length - 1]);

        String line;
        int n = 0;
        while ((line = r.readLine()) != null) {
            if (line.isEmpty()) continue;
            String[] f = line.split(",", -1);
            String row = "golden 第 " + (n + 2) + " 行";
            // 其余下标填 NaN：分组时用错一个下标，和就成了 NaN
            float[] scores = new float[AudioCue.N_CLASSES];
            Arrays.fill(scores, Float.NaN);
            for (int j = 0; j < nIdx; j++) scores[idx[j]] = Float.parseFloat(f[j]);
            float[] got = AudioCue.features(scores);
            for (int g = 0; g < AudioCue.N_FEATURES; g++) {
                float want = Float.parseFloat(f[nIdx + g]);
                assertEquals(row + " " + AudioCue.FEATURE_COLS[g] + " 期望 " + want + " 实际 " + got[g],
                        Float.floatToIntBits(want), Float.floatToIntBits(got[g]));
            }
            assertEquals(row + " 线索", f[f.length - 1], AudioCue.name(AudioCue.classify(got)));
            assertEquals(row + " of()", f[f.length - 1], AudioCue.name(AudioCue.of(scores)));
            n++;
        }
        r.close();
        assertTrue("对拍数据太少: " + n, n >= 1000);
    }

    /** 分组分数 → 线索（只放组里一个下标，组和就是它本身）。 */
    private static int cueOf(float voc, float speech, float music, float silence, float impact) {
        float[] s = new float[AudioCue.N_CLASSES];
        s[11] = voc;           // 尖叫：只在扩展人声组
        s[0] = speech;
        s[132] = music;
        s[494] = silence;
        s[461] = impact;       // 拍打
        return AudioCue.of(s);
    }

    @Test
    public void classifyOrderAndThresholds() {
        assertEquals(AudioCue.DIALOG, cueOf(0f, 0.9f, 0f, 0f, 0f));
        assertEquals(AudioCue.OTHER, cueOf(0.02f, 0.9f, 0f, 0f, 0f));            // 有一点人声：不算对白
        assertEquals(AudioCue.MUSIC, cueOf(0f, 0.1f, 0.5f, 0f, 0f));
        assertEquals(AudioCue.MUSIC, cueOf(0f, 0.4f, 0.4f, 0f, 0f));            // 音乐 = Speech 算音乐
        assertEquals(AudioCue.OTHER, cueOf(0f, 0.45f, 0.4f, 0f, 0f));           // Speech 更高：不是音乐，也不算安静
        assertEquals(AudioCue.QUIET, cueOf(0f, 0.1f, 0.1f, 0.5f, 0f));
        assertEquals(AudioCue.BLIND, cueOf(0f, 0.1f, 0.1f, 0.1f, 0f));
        assertEquals(AudioCue.VOCAL, cueOf(0.5f, 0.1f, 0.5f, 0f, 0f));          // 人声多：不算听不出
        assertEquals(AudioCue.IMPACT, cueOf(0f, 0.1f, 0.5f, 0f, 0.3f));         // 有拍打声：不算听不出
        assertEquals(AudioCue.OTHER, cueOf(0f, 0.1f, 0.5f, 0f, 0.15f));         // 撞击声 0.1~0.2：不听不出也不算拍打
        // 阈值按 double 比：0.7f 比 0.7 小一点，不算对白；大 1 ulp 才算
        assertEquals(AudioCue.OTHER, cueOf(0f, 0.7f, 0f, 0f, 0f));
        assertEquals(AudioCue.DIALOG, cueOf(0f, Math.nextUp(0.7f), 0f, 0f, 0f));
    }

    @Test
    public void unknownWithoutScores() {
        assertNull(AudioCue.features(null));
        assertNull(AudioCue.features(new float[0]));
        assertNull(AudioCue.features(new float[1024]));
        assertEquals(AudioCue.UNKNOWN, AudioCue.of(null));
        assertEquals(AudioCue.UNKNOWN, AudioCue.of(new float[0]));
        assertEquals(AudioCue.UNKNOWN, AudioCue.classify(null));
        assertEquals("", AudioCue.name(AudioCue.UNKNOWN));
        assertEquals("", AudioCue.label(AudioCue.UNKNOWN));
    }

    @Test
    public void namesRoundTrip() {
        for (int c = AudioCue.UNKNOWN; c <= AudioCue.IMPACT; c++) {
            assertEquals(c, AudioCue.fromName(AudioCue.name(c)));
        }
        assertEquals(AudioCue.UNKNOWN, AudioCue.fromName("bogus"));
        assertEquals(AudioCue.UNKNOWN, AudioCue.fromName(null));
        assertEquals("dialog", AudioCue.name(AudioCue.DIALOG));
        assertEquals("对白", AudioCue.label(AudioCue.DIALOG));
    }

    @Test
    public void uninformativeIsMusicQuietBlind() {
        for (int c = AudioCue.UNKNOWN; c <= AudioCue.IMPACT; c++) {
            boolean want = c == AudioCue.MUSIC || c == AudioCue.QUIET || c == AudioCue.BLIND;
            assertEquals(AudioCue.name(c), want, AudioCue.uninformative(c));
        }
    }

    @Test
    public void paramsLineCarriesThresholds() {
        String p = AudioCue.paramsLine();
        assertTrue(p, p.startsWith("CUE=v1 "));
        assertTrue(p, p.contains("CUE_D_VOC=0.0100 CUE_D_SP=0.7000"));
        assertTrue(p, p.contains("CUE_B_IMP=0.1000"));
    }
}
