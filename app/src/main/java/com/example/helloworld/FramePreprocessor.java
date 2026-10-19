package com.example.helloworld;

import java.nio.ByteBuffer;
import java.util.Arrays;

/**
 * 视频模型输入的预处理（纯 Java，不依赖 Android）：把一帧画面变成模型要的 160×160。
 *
 * <p><b>必须和训练时一致。</b>训练素材是这样来的（make_movinet_clips.py、train_mobilenetv3_tap_3class.py）：
 * <pre>
 *   ffmpeg -vf scale=224:224:force_original_aspect_ratio=increase,crop=224:224   短边缩到 224，居中裁 224×224
 *   cv2.resize(frame, (160, 160), interpolation=cv2.INTER_AREA)                 面积平均缩到 160×160
 * </pre>
 * 2026-10-14 之前 app 是把整帧直接压扁成 160×160：16:9 的画面横向被压扁 1.78 倍，训练时裁掉的左右两边也一起送进了模型；
 * 离线模式还是 GPU 从原分辨率一步采样到 160（1080p 时每个输出像素只用到对应约 80 个原像素里的 4 个，有混叠）。
 * 现在三种模式都走这里：
 * <ol>
 *   <li>按 {@link #cropRect} 居中裁正方形（离线模式在 GPU 上裁，网页 / 在线模式在这里裁；在线模式先用 ScreenCropper
 *       去掉截屏里视频四周的黑边，再在视频画面里居中裁）；</li>
 *   <li>面积平均缩到 224（训练片段的尺寸）；</li>
 *   <li>面积平均缩到 160（与训练读帧时的 cv2.INTER_AREA 同一套权重）。</li>
 * </ol>
 *
 * <p>面积缩小与 OpenCV 的 INTER_AREA 是同一种算法（权重表、float 累加顺序、四舍五入到偶数），与 cv2 最多差 1。
 * bench/tools/video_rescore.py 里有逐行对应的 Python 版（离线重算用），改这里要两边一起改，再跑
 * {@code python3 bench/tools/video_rescore.py golden} 重新生成 FramePreprocessorTest 的对拍数据。</p>
 *
 * <p>像素统一用 ARGB 打包的 int（与 Bitmap.getPixels 相同），输出的 alpha 恒为 255。</p>
 */
public final class FramePreprocessor {

    /** 模型输入边长（VideoClassifierHelper.INPUT_SIZE 就是它）。 */
    public static final int MODEL_SIZE = 160;
    /** 训练片段的边长：ffmpeg 先裁成 224×224，训练读帧时再缩到 160。 */
    public static final int CLIP_SIZE = 224;

    /**
     * true：居中裁正方形（与训练一致）；false：整帧压扁（2026-10-14 之前的做法，只留作 A/B 对比）。
     * 改了它，评测 CSV 参数行的 VIDEO_PIPE 会跟着变（见 {@link #geometryName}）。
     */
    public static final boolean CENTER_CROP = true;

    private FramePreprocessor() {}

    /** 写进评测 CSV 参数行 VIDEO_PIPE 的几何部分。 */
    public static String geometryName() {
        return CENTER_CROP ? "crop" : "squash";
    }

    /**
     * 送进模型的区域 {x0, y0, w, h}（像素）。CENTER_CROP 时是居中的正方形，边长等于短边，
     * 与 ffmpeg "scale=...:force_original_aspect_ratio=increase,crop" 取的是同一块（差不到半个像素）；否则是整帧。
     */
    public static int[] cropRect(int w, int h) {
        if (!CENTER_CROP) return new int[]{0, 0, w, h};
        int side = Math.min(w, h);
        return new int[]{(w - side) / 2, (h - side) / 2, side, side};
    }

    /** region {x0, y0, w, h} 里的居中正方形（在线模式：ScreenCropper 找出的视频画面里再居中裁）。 */
    public static int[] cropRect(int[] region) {
        int[] r = cropRect(region[2], region[3]);
        return new int[]{region[0] + r[0], region[1] + r[1], r[2], r[3]};
    }

    /**
     * CPU 阶段：取 region 区域 → 缩到 224×224 → 缩到 160×160。
     *
     * @param argb   源图，ARGB 打包，行优先，宽 w、高 h
     * @param region {x0, y0, w, h}；null 表示整张图（离线模式在 GPU 上已经裁好了）
     * @return 160×160 的 ARGB 像素
     */
    public static int[] toModelInput(int[] argb, int w, int h, int[] region) {
        int x0 = 0, y0 = 0, cw = w, ch = h;
        if (region != null) {
            x0 = region[0];
            y0 = region[1];
            cw = region[2];
            ch = region[3];
        }
        int[] clip = resize(argb, w, x0, y0, cw, ch, CLIP_SIZE, CLIP_SIZE);
        return resize(clip, CLIP_SIZE, 0, 0, CLIP_SIZE, CLIP_SIZE, MODEL_SIZE, MODEL_SIZE);
    }

    /**
     * 把 src 中 {x0, y0, cw, ch} 这块缩放到 dw×dh。两个方向都是缩小时用面积平均（同 cv2.INTER_AREA，
     * 整数倍时走与 OpenCV 相同的整块求和），否则双线性（只在源图比 224 还小时才会用到）。
     *
     * @param stride src 一行的像素数
     */
    static int[] resize(int[] src, int stride, int x0, int y0, int cw, int ch, int dw, int dh) {
        if (cw == dw && ch == dh) {
            int[] out = new int[dw * dh];
            for (int y = 0; y < dh; y++) {
                for (int x = 0; x < dw; x++) out[y * dw + x] = 0xFF000000 | src[(y0 + y) * stride + x0 + x];
            }
            return out;
        }
        if (cw >= dw && ch >= dh) {
            if (cw % dw == 0 && ch % dh == 0) return areaInteger(src, stride, x0, y0, cw, ch, dw, dh);
            return area(src, stride, x0, y0, cw, ch, dw, dh);
        }
        return bilinear(src, stride, x0, y0, cw, ch, dw, dh);
    }

    /** OpenCV computeResizeAreaTab 的权重表：第 k 项表示 dst[di[k]] += src[si[k]] * alpha[k]。 */
    static final class AreaTab {
        final int[] di;
        final int[] si;
        final float[] alpha;
        final int n;

        AreaTab(int[] di, int[] si, float[] alpha, int n) {
            this.di = di;
            this.si = si;
            this.alpha = alpha;
            this.n = n;
        }
    }

    static AreaTab areaTab(int ssize, int dsize) {
        double scale = 1.0 / ((double) dsize / ssize);     // 与 cv::resize 相同：1 / inv_scale
        int cap = ssize * 2 + 2;
        int[] di = new int[cap];
        int[] si = new int[cap];
        float[] alpha = new float[cap];
        int k = 0;
        for (int dx = 0; dx < dsize; dx++) {
            double fsx1 = dx * scale;
            double fsx2 = fsx1 + scale;
            double cellWidth = Math.min(scale, ssize - fsx1);
            int sx1 = (int) Math.ceil(fsx1);
            int sx2 = (int) Math.floor(fsx2);
            sx2 = Math.min(sx2, ssize - 1);
            sx1 = Math.min(sx1, sx2);
            if (sx1 - fsx1 > 1e-3) {
                di[k] = dx;
                si[k] = sx1 - 1;
                alpha[k++] = (float) ((sx1 - fsx1) / cellWidth);
            }
            for (int sx = sx1; sx < sx2; sx++) {
                di[k] = dx;
                si[k] = sx;
                alpha[k++] = (float) (1.0 / cellWidth);
            }
            if (fsx2 - sx2 > 1e-3) {
                di[k] = dx;
                si[k] = sx2;
                alpha[k++] = (float) (Math.min(Math.min(fsx2 - sx2, 1.0), cellWidth) / cellWidth);
            }
        }
        return new AreaTab(di, si, alpha, k);
    }

    /** 非整数倍的面积缩小：与 OpenCV ResizeArea_Invoker 相同的计算顺序（先横向、再按行加权累加，float）。 */
    static int[] area(int[] src, int stride, int x0, int y0, int cw, int ch, int dw, int dh) {
        AreaTab xt = areaTab(cw, dw);
        AreaTab yt = areaTab(ch, dh);
        float[] buf = new float[dw * 3];
        float[] sum = new float[dw * 3];
        int[] out = new int[dw * dh];
        int prevDy = yt.di[0];
        for (int j = 0; j < yt.n; j++) {
            int dy = yt.di[j];
            float beta = yt.alpha[j];
            int row = (y0 + yt.si[j]) * stride + x0;
            Arrays.fill(buf, 0f);
            for (int k = 0; k < xt.n; k++) {
                int p = src[row + xt.si[k]];
                float a = xt.alpha[k];
                int d = xt.di[k] * 3;
                buf[d] = buf[d] + ((p >> 16) & 0xFF) * a;
                buf[d + 1] = buf[d + 1] + ((p >> 8) & 0xFF) * a;
                buf[d + 2] = buf[d + 2] + (p & 0xFF) * a;
            }
            if (dy != prevDy) {
                emitRow(out, prevDy, dw, sum);
                for (int i = 0; i < sum.length; i++) sum[i] = beta * buf[i];
                prevDy = dy;
            } else {
                for (int i = 0; i < sum.length; i++) sum[i] = sum[i] + beta * buf[i];
            }
        }
        emitRow(out, prevDy, dw, sum);
        return out;
    }

    /** 整数倍的面积缩小：整块求和后乘 1/面积（与 OpenCV 的 resizeAreaFast 相同）。 */
    static int[] areaInteger(int[] src, int stride, int x0, int y0, int cw, int ch, int dw, int dh) {
        int fx = cw / dw, fy = ch / dh;
        float scale = 1f / (fx * fy);
        int[] out = new int[dw * dh];
        for (int dy = 0; dy < dh; dy++) {
            for (int dx = 0; dx < dw; dx++) {
                int r = 0, g = 0, b = 0;
                for (int yy = 0; yy < fy; yy++) {
                    int row = (y0 + dy * fy + yy) * stride + x0 + dx * fx;
                    for (int xx = 0; xx < fx; xx++) {
                        int p = src[row + xx];
                        r += (p >> 16) & 0xFF;
                        g += (p >> 8) & 0xFF;
                        b += p & 0xFF;
                    }
                }
                out[dy * dw + dx] = pack(sat(r * scale), sat(g * scale), sat(b * scale));
            }
        }
        return out;
    }

    /** 放大（源图比目标还小时）：像素中心对齐的双线性，double 计算。 */
    static int[] bilinear(int[] src, int stride, int x0, int y0, int cw, int ch, int dw, int dh) {
        int[] out = new int[dw * dh];
        double sx = (double) cw / dw, sy = (double) ch / dh;
        for (int dy = 0; dy < dh; dy++) {
            double fy = (dy + 0.5) * sy - 0.5;
            int ya = (int) Math.floor(fy);
            double wy = fy - ya;
            int y1 = clamp(ya, ch), y2 = clamp(ya + 1, ch);
            for (int dx = 0; dx < dw; dx++) {
                double fx = (dx + 0.5) * sx - 0.5;
                int xa = (int) Math.floor(fx);
                double wx = fx - xa;
                int x1 = clamp(xa, cw), x2 = clamp(xa + 1, cw);
                int p11 = src[(y0 + y1) * stride + x0 + x1], p12 = src[(y0 + y1) * stride + x0 + x2];
                int p21 = src[(y0 + y2) * stride + x0 + x1], p22 = src[(y0 + y2) * stride + x0 + x2];
                int[] c = new int[3];
                for (int i = 0; i < 3; i++) {
                    int sh = 16 - 8 * i;
                    double top = (1 - wx) * ((p11 >> sh) & 0xFF) + wx * ((p12 >> sh) & 0xFF);
                    double bot = (1 - wx) * ((p21 >> sh) & 0xFF) + wx * ((p22 >> sh) & 0xFF);
                    c[i] = sat((1 - wy) * top + wy * bot);
                }
                out[dy * dw + dx] = pack(c[0], c[1], c[2]);
            }
        }
        return out;
    }

    /**
     * glReadPixels(GL_RGBA, GL_UNSIGNED_BYTE) 读出的像素（第一行是图像最下面一行）转成自上而下的 ARGB。
     *
     * @param rgba position 为 0、至少 w*h*4 字节
     */
    public static void rgbaBottomUpToArgb(ByteBuffer rgba, int w, int h, int[] out) {
        byte[] row = new byte[w * 4];                         // 按行批量读：逐字节 get(i) 在 Android 上慢很多
        for (int y = 0; y < h; y++) {
            rgba.position((h - 1 - y) * w * 4);
            rgba.get(row, 0, row.length);
            int dst = y * w;
            for (int x = 0, i = 0; x < w; x++, i += 4) {
                out[dst + x] = pack(row[i] & 0xFF, row[i + 1] & 0xFF, row[i + 2] & 0xFF);
            }
        }
        rgba.rewind();
    }

    private static void emitRow(int[] out, int dy, int dw, float[] sum) {
        int o = dy * dw;
        for (int dx = 0; dx < dw; dx++) {
            out[o + dx] = pack(sat(sum[3 * dx]), sat(sum[3 * dx + 1]), sat(sum[3 * dx + 2]));
        }
    }

    private static int clamp(int i, int n) {
        return i < 0 ? 0 : (i >= n ? n - 1 : i);
    }

    /** 与 OpenCV saturate_cast&lt;uchar&gt;(float) 相同：四舍五入到偶数，再截到 0..255。 */
    private static int sat(double v) {
        int i = (int) Math.rint(v);
        return i < 0 ? 0 : (i > 255 ? 255 : i);
    }

    private static int pack(int r, int g, int b) {
        return 0xFF000000 | (r << 16) | (g << 8) | b;
    }
}
