package com.example.helloworld;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * FramePreprocessor 的单元测试。
 *
 * <p>最重要的是 {@link #matchesPythonGolden()}：bench/tools/video_rescore.py 逐行移植了 FramePreprocessor，
 * 离线重算视频结果、评估换预处理的效果都靠它；两边不一致，离线算出来的数字就不代表手机。
 * 对拍数据由 {@code python3 bench/tools/video_rescore.py golden} 生成；Python 版与 cv2.INTER_AREA 的一致性在
 * {@code video_rescore.py --selftest} 里校验。</p>
 */
public class FramePreprocessorTest {

    /** 与 video_rescore.golden_image 相同的确定性测试图（ARGB）。 */
    private static int[] goldenImage(int w, int h, String pattern, long seed) {
        int[] out = new int[w * h];
        long s = seed;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int r, g, b;
                if ("pattern".equals(pattern)) {
                    r = (x * 7 + y * 3) & 255;
                    g = (x * x + y) & 255;
                    b = ((x ^ y) * 5) & 255;
                } else {
                    s = s * 6364136223846793005L + 1442695040888963407L;
                    r = (int) (s >>> 56);
                    s = s * 6364136223846793005L + 1442695040888963407L;
                    g = (int) (s >>> 56);
                    s = s * 6364136223846793005L + 1442695040888963407L;
                    b = (int) (s >>> 56);
                }
                out[y * w + x] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }
        return out;
    }

    @Test
    public void matchesPythonGolden() throws IOException {
        InputStream in = getClass().getClassLoader().getResourceAsStream("video/preprocess_golden.txt");
        assertNotNull("缺少测试资源 video/preprocess_golden.txt（python3 bench/tools/video_rescore.py golden 生成）", in);
        BufferedReader br = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
        List<String> lines = new ArrayList<>();
        String line;
        while ((line = br.readLine()) != null) lines.add(line);
        lines.add("case end");                          // 哨兵：触发最后一个 case 的校验
        String[] head = null;
        List<String> rows = new ArrayList<>();
        int cases = 0;
        for (String l : lines) {
            if (l.startsWith("#") || l.trim().isEmpty()) continue;
            if (l.startsWith("case")) {
                if (head != null) {
                    check(head, rows);
                    cases++;
                }
                head = l.trim().split("\\s+");
                rows.clear();
                continue;
            }
            rows.add(l.trim());
        }
        assertTrue("对拍数据里的 case 太少", cases >= 8);
    }

    private static void check(String[] f, List<String> rows) {
        String kind = f[1], pattern = f[2];
        long seed = Long.parseLong(f[3]);
        int w = Integer.parseInt(f[4]), h = Integer.parseInt(f[5]);
        int x0 = Integer.parseInt(f[6]), y0 = Integer.parseInt(f[7]);
        int cw = Integer.parseInt(f[8]), ch = Integer.parseInt(f[9]);
        int dw = Integer.parseInt(f[10]), dh = Integer.parseInt(f[11]);
        String tag = kind + " " + w + "x" + h + "->" + dw + "x" + dh;
        int[] img = goldenImage(w, h, pattern, seed);
        int[] got;
        if ("model".equals(kind)) {
            boolean whole = x0 == 0 && y0 == 0 && cw == w && ch == h;
            got = FramePreprocessor.toModelInput(img, w, h, whole ? null : new int[]{x0, y0, cw, ch});
        } else {
            got = FramePreprocessor.resize(img, w, x0, y0, cw, ch, dw, dh);
        }
        assertEquals(tag + " 行数", dh, rows.size());
        assertEquals(tag + " 像素数", dw * dh, got.length);
        for (int y = 0; y < dh; y++) {
            String row = rows.get(y);
            assertEquals(tag + " 第 " + y + " 行长度", dw * 6, row.length());
            for (int x = 0; x < dw; x++) {
                int want = 0xFF000000 | Integer.parseInt(row.substring(x * 6, x * 6 + 6), 16);
                if (want != got[y * dw + x]) {
                    throw new AssertionError(String.format("%s (%d,%d): 期望 %06x，实际 %06x",
                            tag, x, y, want & 0xFFFFFF, got[y * dw + x] & 0xFFFFFF));
                }
            }
        }
    }

    @Test
    public void centerCropMatchesFfmpegCrop() {
        // ffmpeg scale=224:224:force_original_aspect_ratio=increase,crop=224:224 取的就是居中的正方形
        assertArrayEquals(new int[]{420, 0, 1080, 1080}, FramePreprocessor.cropRect(1920, 1080));
        assertArrayEquals(new int[]{280, 0, 720, 720}, FramePreprocessor.cropRect(1280, 720));
        assertArrayEquals(new int[]{0, 420, 1080, 1080}, FramePreprocessor.cropRect(1080, 1920));
        assertArrayEquals(new int[]{0, 0, 640, 640}, FramePreprocessor.cropRect(640, 640));
        assertTrue("默认应当居中裁剪（与训练一致）", FramePreprocessor.CENTER_CROP);
    }

    @Test
    public void constantImageStaysConstant() {
        int c = 0xFF336699;
        for (int[] s : new int[][]{{672, 672}, {540, 1170}, {200, 120}, {1000, 562}}) {
            int[] img = new int[s[0] * s[1]];
            java.util.Arrays.fill(img, c);
            int[] out = FramePreprocessor.toModelInput(img, s[0], s[1], FramePreprocessor.cropRect(s[0], s[1]));
            assertEquals(FramePreprocessor.MODEL_SIZE * FramePreprocessor.MODEL_SIZE, out.length);
            for (int p : out) assertEquals(s[0] + "x" + s[1], c, p);
        }
    }

    @Test
    public void cropKeepsOnlyTheCenter() {
        // 左右两条与中间不同色：居中裁剪后一点都不应混进来；整帧压扁时两边会被缩进画面
        int w = 1280, h = 720;
        int[] img = new int[w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) img[y * w + x] = (x < 280 || x >= 1000) ? 0xFFFF0000 : 0xFF00FF00;
        }
        int[] out = FramePreprocessor.toModelInput(img, w, h, FramePreprocessor.cropRect(w, h));
        for (int p : out) assertEquals(0xFF00FF00, p);
    }

    @Test
    public void rgbaBottomUpIsFlippedToTopDown() {
        int w = 3, h = 2;
        ByteBuffer bb = ByteBuffer.allocate(w * h * 4);
        // glReadPixels 的第一行是图像最下面一行
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                bb.put((byte) (10 * x + y)).put((byte) 200).put((byte) (y == 0 ? 255 : 0)).put((byte) 255);
            }
        }
        bb.rewind();
        int[] out = new int[w * h];
        FramePreprocessor.rgbaBottomUpToArgb(bb, w, h, out);
        for (int x = 0; x < w; x++) {
            assertEquals(0xFF000000 | ((10 * x + 1) << 16) | (200 << 8), out[x]);              // 图像第一行 = 读出的最后一行
            assertEquals(0xFF000000 | ((10 * x) << 16) | (200 << 8) | 255, out[w + x]);
        }
    }
}
