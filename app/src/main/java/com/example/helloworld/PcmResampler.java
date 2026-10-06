package com.example.helloworld;

/**
 * 交错 16-bit PCM → 单声道 float、16kHz，流式。离线模式 AudioDecoder 用它把解码输出变成模型要的音频。
 *
 * <p>早先的 AudioDecoder（已删除）没看声道数，把交错立体声（L R L R …）当单声道，每 4 秒块只用前一半：
 * 双声道片源送进模型的是 0.5 倍速、低八度的音频（bench/tools/audio_rescore.py check 在 6 部片上
 * 证实了这一点，见 bench/README.md）。这里做正确的处理：</p>
 * <ol>
 *   <li>下混：各声道取平均。</li>
 *   <li>重采样：Blackman 窗 sinc 低通插值，截止 {@value #CUTOFF_HZ}Hz（YAMNet 的 mel 上限是 7500Hz），
 *       每侧 {@value #ZERO_CROSSINGS} 个过零点。第 n 个输出样本对应输入里 n/16000 秒处。
 *       输入输出采样率之比是有理数 num/den，插值相位只有 den 种（48k:1 种，44.1k:160 种），系数表预先算好；
 *       每个相位的系数归一化到和为 1。输入就是 16kHz 时直接透传。</li>
 *   <li>输出只按 16 的整数倍给（16 个样本 = 1ms），调用方用 "起点 ms + i*1000/16000" 打时间戳
 *       （PcmCircularBuffer.write）时就与全局连续计时一致，块与块之间不丢样本、不重置相位。</li>
 * </ol>
 *
 * <p>纯 Java、不依赖 Android。bench/tools/audio_rescore.py 的 fixed 管线逐行移植了这里的算法，离线重算出来的
 * 就是手机上会喂给模型的音频；两边由 PcmResamplerTest 对拍（对拍数据用 audio_rescore.py golden 生成）。
 * 改这里必须同步改 Python 端。</p>
 */
public final class PcmResampler {

    public static final int OUT_RATE = 16000;
    static final double CUTOFF_HZ = 7600.0;
    static final int ZERO_CROSSINGS = 24;
    /** 相位数超过这个就不建表，逐个输出现算系数（只有很奇怪的采样率才会走到）。 */
    static final int MAX_TABLE_PHASES = 2048;

    private final int inRate;
    private final int channels;
    private final boolean passthrough;
    private final long num;          // 输出第 n 个样本在输入里的位置 = n * num / den
    private final long den;
    private final int half;          // 用输入 [i0-half+1, i0+half] 这 2*half 个样本
    private final double fc;         // 截止频率，单位：周期/输入样本
    private final double width;      // 窗半宽（输入样本）
    private final double[][] table;  // [相位][2*half]，null = 现算

    private float[] buf = new float[0];
    private long bufStart;           // buf[0] 对应的输入下标（开头补的零下标为负）
    private int bufLen;
    private long inFrames;           // 已收到的输入帧数
    private long outCount;           // 已产出的输出样本数

    public PcmResampler(int inRate, int channels) {
        if (inRate <= 0 || channels <= 0) {
            throw new IllegalArgumentException("inRate=" + inRate + " channels=" + channels);
        }
        this.inRate = inRate;
        this.channels = channels;
        this.passthrough = inRate == OUT_RATE;
        long g = gcd(inRate, OUT_RATE);
        this.num = inRate / g;
        this.den = OUT_RATE / g;
        double cutoff = Math.min(CUTOFF_HZ, 0.475 * Math.min(inRate, OUT_RATE));
        this.fc = cutoff / inRate;
        this.width = ZERO_CROSSINGS / (2.0 * fc);
        this.half = (int) Math.ceil(width);
        if (passthrough) {
            table = null;
        } else if (den <= MAX_TABLE_PHASES) {
            table = new double[(int) den][];
            for (int k = 0; k < den; k++) table[k] = weights(k);
        } else {
            table = null;
        }
        // 开头补 half-1 个零：第 0 个输出要用到下标 -half+1 .. half
        int lead = passthrough ? 0 : half - 1;
        ensure(lead + 1);                                  // 新数组本来就是 0
        bufStart = -lead;
        bufLen = lead;
    }

    /** 相位 k（小数位置 k/den）的 2*half 个系数，第 j 个乘输入下标 i0-half+1+j。 */
    double[] weights(long k) {
        double f = (double) k / den;
        double[] w = new double[2 * half];
        double sum = 0;
        for (int j = 0; j < 2 * half; j++) {
            double t = f + half - 1 - j;                  // 输出位置 - 该输入样本的位置
            double v = 0;
            if (Math.abs(t) < width) {
                double x = 2.0 * fc * t;
                double sinc = x == 0 ? 1.0 : Math.sin(Math.PI * x) / (Math.PI * x);
                double u = t / width;
                double win = 0.42 + 0.5 * Math.cos(Math.PI * u) + 0.08 * Math.cos(2.0 * Math.PI * u);
                v = 2.0 * fc * sinc * win;
            }
            w[j] = v;
            sum += v;
        }
        for (int j = 0; j < w.length; j++) w[j] /= sum;
        return w;
    }

    public int inputRate() { return inRate; }

    public int channelCount() { return channels; }

    /** 已收到的输入帧数（每帧含全部声道）。 */
    public long inputFrames() { return inFrames; }

    /** 已产出的 16kHz 输出样本数；下一次 process 返回的第一个样本就是第 outputCount() 个。总是 16 的倍数。 */
    public long outputCount() { return outCount; }

    /**
     * 喂入 frames 帧交错 PCM（pcm[offset] 起，共 frames*channels 个 short），返回这次能产出的 16kHz 样本
     * （长度为 16 的倍数，可能为 0）。
     */
    public float[] process(short[] pcm, int offset, int frames) {
        ensure(bufLen + frames);
        float scale = channels * 32768f;
        for (int i = 0; i < frames; i++) {
            int sum = 0;
            int base = offset + i * channels;
            for (int c = 0; c < channels; c++) sum += pcm[base + c];
            buf[bufLen++] = sum / scale;
        }
        inFrames += frames;

        long last = inFrames - 1;                          // 已有的最大输入下标
        long avail;                                        // 能算的输出个数（从 0 起计）
        if (passthrough) {
            avail = inFrames;
        } else {
            // 第 n 个输出需要 floor(n*num/den)+half <= last，即 n*num < (last-half+1)*den
            long lim = last - half;
            avail = lim < 0 ? 0 : ((lim + 1) * den - 1) / num + 1;
            while (avail > 0 && ((avail - 1) * num) / den + half > last) avail--;   // 防御：整除边界
        }
        long n1 = avail - (avail % 16);                    // 只产出到 16 的整数倍
        int count = (int) Math.max(0, n1 - outCount);
        float[] out = new float[count];
        for (int m = 0; m < count; m++) {
            long n = outCount + m;
            if (passthrough) {
                out[m] = buf[(int) (n - bufStart)];
                continue;
            }
            long p = n * num;
            long i0 = p / den;
            long k = p % den;
            double[] w = table != null ? table[(int) k] : weights(k);
            int s = (int) (i0 - half + 1 - bufStart);
            double acc = 0;
            for (int j = 0; j < w.length; j++) acc += buf[s + j] * w[j];
            out[m] = (float) acc;
        }
        outCount += count;
        compact();
        return out;
    }

    /** 丢掉以后再也用不到的输入。 */
    private void compact() {
        long keepFrom;
        if (passthrough) {
            keepFrom = outCount;
        } else {
            keepFrom = (outCount * num) / den - half + 1;  // 下一个输出要用的最小下标
        }
        int drop = (int) Math.min(bufLen, Math.max(0, keepFrom - bufStart));
        if (drop > 0) {
            System.arraycopy(buf, drop, buf, 0, bufLen - drop);
            bufLen -= drop;
            bufStart += drop;
        }
    }

    private void ensure(int cap) {
        if (buf.length < cap) {
            float[] nb = new float[Math.max(cap, buf.length * 2 + 1024)];
            System.arraycopy(buf, 0, nb, 0, bufLen);
            buf = nb;
        }
    }

    private static long gcd(long a, long b) {
        while (b != 0) {
            long t = a % b;
            a = b;
            b = t;
        }
        return a;
    }
}
