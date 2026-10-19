package com.example.helloworld;

import java.util.Arrays;

/**
 * 在线模式（录屏）专用：在截屏里找出视频画面的位置（去掉四周的黑边），再交给 FramePreprocessor 居中裁正方形。
 * 纯 Java，不依赖 Android。
 *
 * <p>训练素材是视频文件本身的画面取居中正方形；在线模式拿到的是整个屏幕。只有视频把屏幕的短边占满（横屏全屏）时，
 * "屏幕的居中正方形"才等于"视频的居中正方形"。其余情况视频四周是黑边，直接取屏幕的居中正方形会把黑边一起裁进来，
 * 视频被缩小（2400×1080 的手机、16:9 的视频）：
 * <ul>
 *   <li>竖屏全屏播放：视频是屏幕中间的一条，在居中正方形里只占 56% 的高度，上下是黑边；</li>
 *   <li>截图尺寸没跟上转屏（OnlineAnalysisService 收不到转屏通知的设备）：系统把横屏画面等比缩进竖屏尺寸的截图、
 *       上下补黑，整幅视频只占居中正方形中间 45% 高的一条；</li>
 *   <li>横屏看竖屏视频：左右是大黑边，居中正方形里视频只占中间 56% 的宽度。</li>
 * </ul>
 *
 * <p>做法：播放器、系统补的黑边是纯黑的，而且上下（左右）对称。按行数"不黑"的像素，找出最长的一段连续亮着的行；
 * 两边剩下的黑边一样宽（差不超过 max(2, 高/50)）才当黑边裁掉，不对称（画面上沿暗、贴着视频的控制条）就不裁。
 * 列同样处理（只数留下的那些行）。最长的一段太短（黑场、只有黑边里的字幕或加载圈）这一帧不作数，沿用上一次的结果。
 * 裁过一次以后，只要亮着的范围没超出上次裁剩的范围（暗场景、视频边上暗下去）就沿用，不跟着缩：
 * 黑边不会自己变亮，范围变大了才说明画面的位置真的变了。</p>
 *
 * <p>局限：片源自己带的黑边（宽银幕压在 16:9 里）也是对称的纯黑，同样会被裁掉，比训练（连黑边一起居中裁）放大约
 * 1.3 倍；背景不是纯黑（如 #121212）的播放器不裁，退回整个截屏的居中正方形。</p>
 *
 * <p>bench/tools/video_rescore.py 里有逐行对应的 Python 版（离线模拟在线模式时用），改这里要两边一起改，再跑
 * {@code python3 bench/tools/video_rescore.py golden} 重新生成 ScreenCropperTest 的对拍数据。</p>
 */
public final class ScreenCropper {

    /** false：不去黑边，直接取整个截屏的居中正方形（2026-10-19 之前的做法，A/B 对比用）。 */
    public static final boolean ENABLED = true;

    /** R、G、B 都不超过它算黑。播放器、系统补的黑边是 0。 */
    static final int BLACK_MAX = 16;
    /** 一行（列）里不黑的像素 ≥ 长度 / LIT_DIV（至少 1 个）才算这一行（列）亮着：黑边里零星的亮点不算。 */
    static final int LIT_DIV = 64;
    /** 最长的一段亮着的行（列）不到这一边的 15%：这一帧不作数（竖屏看宽银幕约 19%）。 */
    static final int MIN_KEEP_PCT = 15;
    /** 两边的黑边宽度差不超过 max(2, 长度 / SYM_DIV) 才当黑边裁掉。 */
    static final int SYM_DIV = 50;

    private int width = -1, height = -1;
    private int[] rowCnt = new int[0], colCnt = new int[0];
    /** 上一次裁掉黑边后留下的行 / 列范围 {a, b}；null = 没裁。 */
    private int[] rows, cols;
    private int[] last;

    /** 清空（下一帧从头找）。截图尺寸变了时 update 会自己清。 */
    public void reset() {
        width = height = -1;
    }

    /**
     * 看一帧截屏，返回视频画面的位置 {x0, y0, w, h}（没有黑边就是整张图）。按时间顺序喂同一路截屏。
     *
     * @param argb 截屏像素，ARGB 打包，行优先，宽 w、高 h
     */
    public int[] update(int[] argb, int w, int h) {
        if (w != width || h != height) {
            width = w;
            height = h;
            rowCnt = new int[h];
            colCnt = new int[w];
            rows = cols = null;
            last = new int[]{0, 0, w, h};
        }
        for (int y = 0; y < h; y++) {
            int n = 0;
            for (int i = y * w, end = i + w; i < end; i++) {
                if (lit(argb[i])) n++;
            }
            rowCnt[y] = n;
        }
        int[] run = longestRun(rowCnt, h, Math.max(1, w / LIT_DIV));
        if (run == null || (run[1] - run[0]) * 100 < MIN_KEEP_PCT * h) return last.clone();
        rows = choose(run, h, rows);
        int y0 = rows == null ? 0 : rows[0], y1 = rows == null ? h : rows[1];

        Arrays.fill(colCnt, 0);
        for (int y = y0; y < y1; y++) {
            for (int x = 0, i = y * w; x < w; x++, i++) {
                if (lit(argb[i])) colCnt[x]++;
            }
        }
        run = longestRun(colCnt, w, Math.max(1, (y1 - y0) / LIT_DIV));
        if (run != null && (run[1] - run[0]) * 100 >= MIN_KEEP_PCT * w) cols = choose(run, w, cols);
        int x0 = cols == null ? 0 : cols[0], x1 = cols == null ? w : cols[1];
        last = new int[]{x0, y0, x1 - x0, y1 - y0};
        return last.clone();
    }

    static boolean lit(int p) {
        return ((p >> 16) & 0xFF) > BLACK_MAX || ((p >> 8) & 0xFF) > BLACK_MAX || (p & 0xFF) > BLACK_MAX;
    }

    /** cnt[0..n) 里最长的一段连续 ≥ need 的 → {a, b}，一样长取前面的；没有返回 null。 */
    static int[] longestRun(int[] cnt, int n, int need) {
        int bestA = 0, bestLen = 0, a = -1;
        for (int i = 0; i <= n; i++) {
            boolean on = i < n && cnt[i] >= need;
            if (on && a < 0) {
                a = i;
            } else if (!on && a >= 0) {
                if (i - a > bestLen) {
                    bestA = a;
                    bestLen = i - a;
                }
                a = -1;
            }
        }
        return bestLen == 0 ? null : new int[]{bestA, bestA + bestLen};
    }

    /** 这一边留下的范围 {a, b}，null = 不裁。亮着的范围没超出上次裁剩的 kept 就沿用；否则两边的黑边对称才裁。 */
    static int[] choose(int[] run, int n, int[] kept) {
        int a = run[0], b = run[1];
        if (kept != null && a >= kept[0] && b <= kept[1]) return kept;
        if (b - a < n && Math.abs(a - (n - b)) <= Math.max(2, n / SYM_DIV)) return run;
        return null;
    }
}
