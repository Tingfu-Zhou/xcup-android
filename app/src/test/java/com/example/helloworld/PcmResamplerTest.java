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
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * PcmResampler 的单元测试。
 *
 * <p>最重要的是 {@link #matchesPythonGolden()}：bench/tools/audio_rescore.py 的 fixed 管线逐行移植了
 * PcmResampler，离线重算出的 run CSV 用来拟合融合配置；两边不一致，拟合出的配置装到手机上就不是那个
 * 效果。对拍数据由 {@code python3 bench/tools/audio_rescore.py golden} 生成。</p>
 */
public class PcmResamplerTest {

    /** 与 audio_rescore.golden_pcm 相同的 64 位 LCG 白噪声。 */
    private static short[] goldenPcm(int sr, int ch, int frames) {
        short[] out = new short[frames * ch];
        long s = sr * 31L + ch;
        for (int i = 0; i < out.length; i++) {
            s = s * 6364136223846793005L + 1442695040888963407L;
            out[i] = (short) ((s >>> 48) - 32768);
        }
        return out;
    }

    /** 按随机长度分块喂入（模拟 MediaCodec 一次给一个 buffer），把输出拼起来。 */
    private static float[] runChunked(PcmResampler r, short[] pcm, int ch, long seed) {
        Random rnd = new Random(seed);
        int frames = pcm.length / ch;
        List<float[]> parts = new ArrayList<>();
        int total = 0;
        for (int f = 0; f < frames; ) {
            int n = Math.min(frames - f, 1 + rnd.nextInt(1500));
            float[] o = r.process(pcm, f * ch, n);
            assertEquals("每次产出都应是 16 的整数倍", 0, o.length % 16);
            parts.add(o);
            total += o.length;
            f += n;
        }
        float[] all = new float[total];
        int k = 0;
        for (float[] o : parts) {
            System.arraycopy(o, 0, all, k, o.length);
            k += o.length;
        }
        assertEquals(total, r.outputCount());
        return all;
    }

    @Test
    public void matchesPythonGolden() throws IOException {
        InputStream in = getClass().getClassLoader().getResourceAsStream("audio/resampler_golden.txt");
        assertNotNull("缺少测试资源 audio/resampler_golden.txt（python3 bench/tools/audio_rescore.py golden 生成）", in);
        BufferedReader br = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
        String line;
        int cases = 0;
        int sr = 0, ch = 0, frames = 0, nOut = 0;
        List<Float> want = new ArrayList<>();
        List<String> lines = new ArrayList<>();
        while ((line = br.readLine()) != null) lines.add(line);
        lines.add("case 0 0 0 0");                      // 哨兵：触发最后一个 case 的校验
        for (String l : lines) {
            if (l.startsWith("#") || l.trim().isEmpty()) continue;
            if (l.startsWith("case")) {
                if (sr > 0) {
                    check(sr, ch, frames, nOut, want);
                    cases++;
                }
                String[] f = l.trim().split("\\s+");
                sr = Integer.parseInt(f[1]);
                ch = Integer.parseInt(f[2]);
                frames = Integer.parseInt(f[3]);
                nOut = Integer.parseInt(f[4]);
                want.clear();
                continue;
            }
            for (String v : l.trim().split("\\s+")) want.add(Float.parseFloat(v));
        }
        assertTrue("对拍数据里没有 case", cases >= 5);
    }

    private static void check(int sr, int ch, int frames, int nOut, List<Float> want) {
        String tag = sr + "Hz/" + ch + "ch";
        assertEquals(tag + " 对拍数据行数", nOut, want.size());
        float[] got = runChunked(new PcmResampler(sr, ch), goldenPcm(sr, ch, frames), ch, sr + ch);
        assertEquals(tag + " 输出个数", nOut, got.length);
        for (int i = 0; i < nOut; i++) {
            // Java 与 numpy 的 sin/cos 可能差 1ulp，系数随之差一点点；累加顺序两边相同
            assertEquals(tag + " 第 " + i + " 个样本", want.get(i), got[i], 1e-6f);
        }
    }

    @Test
    public void chunkingDoesNotChangeOutput() {
        short[] pcm = goldenPcm(44100, 2, 44100);
        float[] once = new PcmResampler(44100, 2).process(pcm, 0, pcm.length / 2);
        float[] chunked = runChunked(new PcmResampler(44100, 2), pcm, 2, 7);
        assertEquals(once.length, chunked.length);
        for (int i = 0; i < once.length; i++) {
            assertEquals("分块喂入与一次喂入应逐位相同 @" + i, once[i], chunked[i], 0f);
        }
    }

    /** 输出第 n 个样本对应输入的 n/16000 秒：1kHz 正弦重采样后应几乎原样（幅度、相位），10kHz 应被滤掉。 */
    @Test
    public void filterKeepsPassbandAndRemovesAliases() {
        for (int sr : new int[]{48000, 44100}) {
            double lowRms = toneRms(sr, 1000.0, true);
            double highRms = toneRms(sr, 10000.0, false);
            assertTrue(sr + "Hz: 1kHz 应基本无损，误差 rms=" + lowRms, lowRms < 1e-3);
            assertTrue(sr + "Hz: 10kHz 应被压到 -60dB 以下，rms=" + highRms, highRms < 0.5 * 1e-3);
        }
    }

    /** checkMatch=true 返回与理想 1kHz 正弦的误差 rms；否则返回输出本身的 rms。幅度 0.5。 */
    private static double toneRms(int sr, double hz, boolean checkMatch) {
        int frames = sr;                                   // 1 秒
        short[] pcm = new short[frames * 2];
        for (int i = 0; i < frames; i++) {
            short v = (short) Math.round(16384 * Math.sin(2 * Math.PI * hz * i / sr));
            pcm[2 * i] = v;
            pcm[2 * i + 1] = v;
        }
        float[] y = new PcmResampler(sr, 2).process(pcm, 0, frames);
        double acc = 0;
        int n = 0;
        for (int i = 800; i < y.length - 800; i++) {      // 跳过开头补零造成的过渡
            double ref = checkMatch ? 0.5 * Math.sin(2 * Math.PI * hz * i / 16000.0) : 0;
            acc += (y[i] - ref) * (y[i] - ref);
            n++;
        }
        return Math.sqrt(acc / n);
    }

    @Test
    public void passthroughAt16kOnlyDownmixes() {
        short[] pcm = {100, 300, -200, 200, 32767, 32767};
        PcmResampler r = new PcmResampler(16000, 2);
        float[] y = r.process(pcm, 0, 3);
        assertEquals("不足 16 个不产出", 0, y.length);
        short[] more = new short[2 * 13];
        y = r.process(more, 0, 13);
        assertEquals(16, y.length);
        assertEquals(400 / 65536f, y[0], 0f);
        assertEquals(0f, y[1], 0f);
        assertEquals(65534 / 65536f, y[2], 0f);
    }
}
