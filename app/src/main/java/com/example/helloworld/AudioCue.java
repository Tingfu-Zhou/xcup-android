package com.example.helloworld;

import java.util.Locale;

/**
 * 音频线索：用 YAMNet 自带的 AudioSet 521 类分数（tflite 第 0 个输出，各帧平均）判断"音频这一刻靠不靠得住"。
 * 每个音频结果判一个线索，交给 {@link FusionArbiter}：
 * <ul>
 *   <li>{@link #DIALOG} 对白：非语言人声组很低、Speech 高。剧情里分类头这时判"转"多半是把说话当成了动作声；
 *       仲裁器在视频确定是剧情时不让它启动马达。</li>
 *   <li>{@link #MUSIC} 音乐 / {@link #QUIET} 安静 / {@link #BLIND} 听不出：没有人声、没有撞击/口腔声，分类头
 *       判什么都不可信（动作被背景音乐盖住时它常判"不转"）。仲裁器在音频连续听不出、视频又很确定时让视频单独启动。</li>
 *   <li>{@link #VOCAL} 人声 / {@link #IMPACT} 拍打 / {@link #OTHER} 其它：信息量正常，规则不看。</li>
 *   <li>{@link #UNKNOWN} 未知：没有分数（模型没有第 0 个输出、推理失败）。仲裁器的线索规则都不生效。</li>
 * </ul>
 *
 * <p>类别组按含义定（下标见 bench/tools/yamnet_class_map.csv），阈值是 2026-10-10/11 在 13 部片上用留一视频和
 * 新片检验后定的（bench/README.md 第 6 节，bench/tools/cue_lab.py 的 PRESETS['a1bvid']）。
 * 分组求和按下标顺序做 float 累加，阈值用 double 比较：与 cue_lab.py / fusion_lab.py 逐位一致
 * （AudioCueTest 用 fusion_lab.py golden 生成的对拍数据校验）。改这里要同步改 fusion_lab.py 的 cue_features /
 * audio_cue，并重新生成对拍数据。</p>
 *
 * <p>纯 Java、不依赖 Android。</p>
 */
public final class AudioCue {

    private AudioCue() {
    }

    public static final int UNKNOWN = 0;
    public static final int OTHER = 1;
    public static final int DIALOG = 2;
    public static final int MUSIC = 3;
    public static final int QUIET = 4;
    public static final int BLIND = 5;
    public static final int VOCAL = 6;
    public static final int IMPACT = 7;

    /** 评测 CSV a_cue 列的写法（与 cue_lab.py 的线索名一致），UNKNOWN 写空串。 */
    private static final String[] NAMES = {"", "other", "dialog", "music", "quiet", "blind", "vocal", "impact"};
    private static final String[] LABELS = {"", "其它", "对白", "音乐", "安静", "听不出", "人声", "拍打"};

    /** AudioSet 类别数（YAMNet 第 0 个输出的最后一维）。 */
    public static final int N_CLASSES = 521;

    // ======================================================== 类别组（与 cue_lab.py 的 GROUPS 顺序、内容相同）
    /** 非语言人声核心组：呜咽 21、呻吟 22、叹气 23、哼 33、咕哝 34、呼吸 36、倒吸气 39、喘 40。只记录，规则不用。 */
    static final int[] VOC_CORE = {21, 22, 23, 33, 34, 36, 39, 40};
    /** 非语言人声扩展组 = 核心组 + 尖叫 11、笑 13/15/16/17/18、哭 19。规则用这一组。 */
    static final int[] VOC_EXT = {11, 13, 15, 16, 17, 18, 19, 21, 22, 23, 33, 34, 36, 39, 40};
    static final int[] SPEECH = {0};
    /** Speech + Conversation 2 + Narration 3。只记录，规则不用。 */
    static final int[] SPEECH_EXT = {0, 2, 3};
    static final int[] MUSIC_G = {132};
    static final int[] SILENCE_G = {494};
    /** 撞击声和口腔声：咀嚼 49、敲 353、轻敲 354、挤压水声 441、拍打 461、重击 462。有它们就不算"听不出"。 */
    static final int[] IMPACT_G = {49, 353, 354, 441, 461, 462};

    /** {@link #features} 返回数组的下标（也是评测 CSV a_cue_* 列的顺序）。 */
    public static final int F_VOC_CORE = 0, F_VOC_EXT = 1, F_SPEECH = 2, F_SPEECH_EXT = 3,
            F_MUSIC = 4, F_SILENCE = 5, F_IMPACT = 6, N_FEATURES = 7;
    private static final int[][] GROUPS = {VOC_CORE, VOC_EXT, SPEECH, SPEECH_EXT, MUSIC_G, SILENCE_G, IMPACT_G};
    /** 评测 CSV 的列名（与 cue_lab.py 的 GROUPS 名字对应；BenchLogger 的表头逐字写了同样的名字）。
     *  不用 a_ys_ 前缀：那是 audio_rescore.py 离线写的单个类别分数，四位小数，规则要全精度的。 */
    public static final String[] FEATURE_COLS = {"a_cue_voc_core", "a_cue_voc_ext", "a_cue_speech",
            "a_cue_speech_ext", "a_cue_music", "a_cue_silence", "a_cue_impact"};

    // ======================================================== 阈值
    /** 对白：人声组 < D_VOC 且 Speech ≥ D_SP。 */
    static final double D_VOC = 0.01;
    static final double D_SP = 0.7;
    /** 听不出：人声组 < B_VOC、撞击/口腔声 < B_IMP；Music ≥ B_MUS 且不低于 Speech 为音乐，
     *  否则 Speech < B_SP 时 Silence ≥ B_SIL 为安静、都不是为听不出。 */
    static final double B_VOC = 0.05;
    static final double B_SP = 0.2;
    static final double B_IMP = 0.1;
    static final double B_MUS = 0.3;
    static final double B_SIL = 0.3;
    /** 人声组 ≥ P_VOC 为人声；撞击/口腔声 ≥ P_IMP 为拍打（只记录）。 */
    static final double P_VOC = 0.3;
    static final double P_IMP = 0.2;

    /**
     * 521 类分数 → 各组分数（{@link #N_FEATURES} 个，按 F_* 下标）。每组按下标顺序 float 累加。
     *
     * @return 分数不是 521 维时返回 null（线索为 UNKNOWN）
     */
    public static float[] features(float[] scores) {
        if (scores == null || scores.length != N_CLASSES) return null;
        float[] f = new float[N_FEATURES];
        for (int g = 0; g < GROUPS.length; g++) {
            float acc = 0f;
            for (int idx : GROUPS[g]) acc += scores[idx];
            f[g] = acc;
        }
        return f;
    }

    /** 各组分数 → 线索。按顺序判：对白 → 听不出三种 → 人声 → 拍打 → 其它。f 为 null 时 UNKNOWN。 */
    public static int classify(float[] f) {
        if (f == null || f.length < N_FEATURES) return UNKNOWN;
        double voc = f[F_VOC_EXT], sp = f[F_SPEECH], mus = f[F_MUSIC], sil = f[F_SILENCE], imp = f[F_IMPACT];
        if (voc < D_VOC && sp >= D_SP) return DIALOG;
        if (voc < B_VOC && imp < B_IMP) {
            if (mus >= B_MUS && mus >= sp) return MUSIC;
            if (sp < B_SP) return sil >= B_SIL ? QUIET : BLIND;
        }
        if (voc >= P_VOC) return VOCAL;
        if (imp >= P_IMP) return IMPACT;
        return OTHER;
    }

    /** 521 类分数直接判线索。 */
    public static int of(float[] scores) {
        return classify(features(scores));
    }

    /** 听不出：音乐 / 安静 / 听不出（仲裁器"音频听不出时视频单独启动"用）。 */
    public static boolean uninformative(int cue) {
        return cue == MUSIC || cue == QUIET || cue == BLIND;
    }

    /** 评测 CSV 的写法（"dialog" 等），UNKNOWN 为空串。 */
    public static String name(int cue) {
        return cue >= 0 && cue < NAMES.length ? NAMES[cue] : "";
    }

    /** a_cue 列 → 线索；空串或认不出的为 UNKNOWN。 */
    public static int fromName(String s) {
        if (s == null || s.isEmpty()) return UNKNOWN;
        for (int i = 1; i < NAMES.length; i++) {
            if (NAMES[i].equals(s)) return i;
        }
        return UNKNOWN;
    }

    /** 界面上显示的中文名，UNKNOWN 为空串。 */
    public static String label(int cue) {
        return cue >= 0 && cue < LABELS.length ? LABELS[cue] : "";
    }

    /** 写进评测 CSV 参数行：线索的阈值。 */
    public static String paramsLine() {
        return String.format(Locale.US,
                "CUE=v1 CUE_D_VOC=%.4f CUE_D_SP=%.4f CUE_B_VOC=%.4f CUE_B_SP=%.4f CUE_B_IMP=%.4f "
                        + "CUE_B_MUS=%.4f CUE_B_SIL=%.4f CUE_P_VOC=%.4f CUE_P_IMP=%.4f",
                D_VOC, D_SP, B_VOC, B_SP, B_IMP, B_MUS, B_SIL, P_VOC, P_IMP);
    }
}
