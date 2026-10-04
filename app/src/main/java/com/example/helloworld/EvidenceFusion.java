package com.example.helloworld;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/**
 * 证据融合：把两个模型的原始概率换算成"证据"（对数似然比 LLR），随时间累积，双阈值迟滞判定 转/不转。
 *
 * <p>为什么替换原来的投票窗 + 启动仲裁：那套规则把"音频更准、音频权重 1.2、音频可直通启动"写死在
 * 代码里，bench 实测这个前提不成立（音频在剧情段也大量判"转"），而且每换一次模型就得重写规则。
 * 这里每个模态"说话算多少"由 bench 统计出的证据表决定：换了模型，用
 * {@code bench/tools/fusion_lab.py fit} 重新生成 {@code assets/fusion_config.json} 即可，不用改代码。</p>
 *
 * <pre>
 * 证据：模型每出一个新结果，取特征 s = Σ weights[i]·probs[i]（裁到 [0,1]），按 edges 分箱查 llr，
 *       llr = log P(s 落在该区间 | 动作) / P(s 落在该区间 | 剧情)
 * 累积：L = clamp(L + wV·llrV + wA·llrA, -clamp, +clamp)        只在有新结果时更新
 * 决策：不转 → 转：L ≥ thetaOn；转 → 不转：L ≤ thetaOff        thetaOff &lt; thetaOn，迟滞
 * </pre>
 *
 * <p>每个模型结果只计一次（按结果时间戳去重）。融合循环 500ms 一次而音频 1000ms 才出一个结果，
 * 原来的投票窗会把同一个音频结果数两遍。过期（超过 maxAgeMs）或推理失败的结果不给证据。</p>
 *
 * <p><b>与 bench/tools/fusion_lab.py 里的 EvidenceFusion 逐 tick 对拍</b>：全部 double 运算、加法顺序一致，
 * 对拍数据 {@code app/src/test/resources/fusion/golden.csv} 由 EvidenceFusionTest 校验。改这里的算法，
 * 必须同步改 Python 端并用 {@code fusion_lab.py golden} 重新生成对拍数据。</p>
 *
 * <p>除 org.json（Android 自带）外不依赖 Android。只在融合线程调用，不加锁。</p>
 */
public final class EvidenceFusion {

    /** 不运行。 */
    public static final String MODE_LEGACY = "legacy";
    /** 每 tick 计算并写进评测 CSV，但蓝牙仍由原融合驱动。 */
    public static final String MODE_SHADOW = "shadow";
    /** 由证据融合驱动蓝牙。 */
    public static final String MODE_LIVE = "live";

    public static final String FORMAT = "xcup-fusion-v1";

    private static final int NONE = 0;
    private static final int NOISE = 1;
    private static final int DO = 2;

    /** 一个模型结果：原始概率 + 产出时刻（墙钟毫秒）。probs 为 null 表示推理失败。不可变，可跨线程传递。 */
    public static final class Snapshot {
        public final float[] probs;
        public final long tsMs;

        public Snapshot(float[] probs, long tsMs) {
            this.probs = probs;
            this.tsMs = tsMs;
        }
    }

    /** 一个模态的证据表：特征权重（对应模型输出的类目顺序）+ 分箱边界 + 每格的 LLR。 */
    public static final class Modality {
        final double[] weights;
        final double[] edges;
        final double[] llr;

        public Modality(double[] weights, double[] edges, double[] llr) {
            if (weights == null || weights.length == 0) {
                throw new IllegalArgumentException("weights 为空");
            }
            if (edges == null || edges.length < 2 || llr == null || llr.length != edges.length - 1) {
                throw new IllegalArgumentException("edges 与 llr 长度不匹配");
            }
            for (int i = 0; i + 1 < edges.length; i++) {
                if (!(edges[i] < edges[i + 1])) {
                    throw new IllegalArgumentException("edges 必须严格递增");
                }
            }
            for (double x : llr) {
                if (Double.isNaN(x) || Double.isInfinite(x)) {
                    throw new IllegalArgumentException("llr 含 NaN/Inf");
                }
            }
            this.weights = weights.clone();
            this.edges = edges.clone();
            this.llr = llr.clone();
        }

        /** 推理失败时 Java 端给的是 null / 空数组 / 全 0 数组，一律不给证据。 */
        boolean valid(float[] probs) {
            if (probs == null || probs.length < weights.length) return false;
            double sum = 0;
            for (float p : probs) {
                if (Float.isNaN(p) || Float.isInfinite(p)) return false;
                sum += p;
            }
            return sum > 0.5;
        }

        double feature(float[] probs) {
            double s = 0;
            for (int i = 0; i < weights.length; i++) {
                s += weights[i] * probs[i];
            }
            return s < 0 ? 0 : (s > 1 ? 1 : s);
        }

        double lookup(double s) {
            int i = 0;
            for (int k = 1; k < edges.length - 1; k++) {
                if (s >= edges[k]) {
                    i = k;
                } else {
                    break;
                }
            }
            return llr[i];
        }
    }

    /** assets/fusion_config.json 的内容。 */
    public static final class Config {
        public final Modality video;
        public final Modality audio;
        public final double wV;
        public final double wA;
        public final double clamp;
        public final double thetaOn;
        public final double thetaOff;
        public final long maxAgeMs;
        public final String mode;
        /** 写进评测 CSV 的参数行，追溯某份 dump 是哪份配置跑出来的。 */
        public final String id;
        /** 未经回放验证的临时配置（例如手算的证据表）。 */
        public final boolean provisional;

        public Config(Modality video, Modality audio, double wV, double wA, double clamp,
                      double thetaOn, double thetaOff, long maxAgeMs, String mode, String id,
                      boolean provisional) {
            if (video == null || audio == null) {
                throw new IllegalArgumentException("缺少 video/audio 证据表");
            }
            if (!(clamp > 0 && -clamp < thetaOff && thetaOff < thetaOn && thetaOn <= clamp)) {
                throw new IllegalArgumentException("需满足 -clamp < theta_off < theta_on <= clamp");
            }
            if (!MODE_LEGACY.equals(mode) && !MODE_SHADOW.equals(mode) && !MODE_LIVE.equals(mode)) {
                throw new IllegalArgumentException("mode 只能是 legacy/shadow/live: " + mode);
            }
            this.video = video;
            this.audio = audio;
            this.wV = wV;
            this.wA = wA;
            this.clamp = clamp;
            this.thetaOn = thetaOn;
            this.thetaOff = thetaOff;
            this.maxAgeMs = maxAgeMs;
            this.mode = mode;
            this.id = id;
            this.provisional = provisional;
        }

        /** 解析失败抛 JSONException / IllegalArgumentException，调用方应回退到原融合。 */
        public static Config fromJson(String text) throws JSONException {
            JSONObject o = new JSONObject(text);
            if (!FORMAT.equals(o.optString("format"))) {
                throw new JSONException("不认识的配置格式: " + o.optString("format"));
            }
            return new Config(modality(o.getJSONObject("video")), modality(o.getJSONObject("audio")),
                    o.getDouble("w_v"), o.getDouble("w_a"), o.getDouble("clamp"),
                    o.getDouble("theta_on"), o.getDouble("theta_off"),
                    o.optLong("max_age_ms", 2000), o.optString("mode", MODE_SHADOW),
                    o.optString("id", "unnamed"), o.optBoolean("provisional", false));
        }

        private static Modality modality(JSONObject m) throws JSONException {
            return new Modality(doubles(m.getJSONArray("weights")), doubles(m.getJSONArray("edges")),
                    doubles(m.getJSONArray("llr")));
        }

        private static double[] doubles(JSONArray a) throws JSONException {
            double[] d = new double[a.length()];
            for (int i = 0; i < d.length; i++) {
                d[i] = a.getDouble(i);
            }
            return d;
        }
    }

    /** 一个 tick 的结果。action 为 "do" / "Noise"，还没收到任何有效结果时为 ""。llr 未使用时为 NaN。 */
    public static final class Output {
        public final String action;
        public final double l;
        public final double llrV;
        public final double llrA;

        Output(String action, double l, double llrV, double llrA) {
            this.action = action;
            this.l = l;
            this.llrV = llrV;
            this.llrA = llrA;
        }
    }

    private final Config cfg;
    private double l;
    private int state;
    private long lastVideoTs;
    private long lastAudioTs;

    public EvidenceFusion(Config cfg) {
        if (cfg == null) throw new IllegalArgumentException("cfg 为空");
        this.cfg = cfg;
        reset();
    }

    public Config config() {
        return cfg;
    }

    /** seek / 静音重置时调用：清空累积证据，回到"无判定"。 */
    public void reset() {
        l = 0;
        state = NONE;
        lastVideoTs = Long.MIN_VALUE;
        lastAudioTs = Long.MIN_VALUE;
    }

    /**
     * 每个融合 tick 调一次。
     *
     * @param video 视频模型最近一个结果（可为 null）
     * @param audio 音频模型最近一个结果（可为 null）
     * @param nowMs 当前墙钟
     */
    public Output update(Snapshot video, Snapshot audio, long nowMs) {
        double llrV = Double.NaN;
        double llrA = Double.NaN;
        if (video != null && video.tsMs > lastVideoTs) {
            lastVideoTs = video.tsMs;
            if (nowMs - video.tsMs <= cfg.maxAgeMs && cfg.video.valid(video.probs)) {
                llrV = cfg.video.lookup(cfg.video.feature(video.probs));
            }
        }
        if (audio != null && audio.tsMs > lastAudioTs) {
            lastAudioTs = audio.tsMs;
            if (nowMs - audio.tsMs <= cfg.maxAgeMs && cfg.audio.valid(audio.probs)) {
                llrA = cfg.audio.lookup(cfg.audio.feature(audio.probs));
            }
        }
        boolean usedV = !Double.isNaN(llrV);
        boolean usedA = !Double.isNaN(llrA);
        if (usedV || usedA) {
            double d = 0;
            if (usedV) d += cfg.wV * llrV;
            if (usedA) d += cfg.wA * llrA;
            double x = l + d;
            l = x > cfg.clamp ? cfg.clamp : (x < -cfg.clamp ? -cfg.clamp : x);
            if (state == DO) {
                if (l <= cfg.thetaOff) state = NOISE;
            } else {
                state = l >= cfg.thetaOn ? DO : NOISE;
            }
        }
        return new Output(state == NONE ? "" : (state == DO ? "do" : "Noise"), l, llrV, llrA);
    }
}
