package com.example.helloworld;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

/**
 * ScreenCropper 的单元测试。
 *
 * <p>{@link #matchesPythonGolden()}：bench/tools/video_rescore.py 逐行移植了 ScreenCropper，离线模拟在线模式（exact_scr…bar
 * 管线）靠它；两边不一致，离线算出来的数字就不代表手机。对拍数据由 {@code python3 bench/tools/video_rescore.py golden}
 * 生成；每个场景该得到什么结果（裁不裁、沿用不沿用）在 {@code video_rescore.py --selftest} 里另外核对。</p>
 */
public class ScreenCropperTest {

    /** 与 video_rescore.crop_golden_frame 相同的测试图：先全黑，再按顺序画矩形 "x0,y0,w,h,填充"。 */
    private static int[] frame(int w, int h, String[] rects, int from) {
        int[] img = new int[w * h];
        Arrays.fill(img, 0xFF000000);
        for (int k = from; k < rects.length; k++) {
            String[] f = rects[k].split(",");
            int x0 = Integer.parseInt(f[0]), y0 = Integer.parseInt(f[1]);
            int cw = Integer.parseInt(f[2]), ch = Integer.parseInt(f[3]);
            String fill = f[4];
            char kind = fill.charAt(0);
            long s = kind == 'n' ? Long.parseLong(fill.substring(1)) : 0;
            int color = kind == 'c' ? 0xFF000000 | Integer.parseInt(fill.substring(1), 16) : 0;
            int step = kind == 'k' ? Integer.parseInt(fill.substring(1)) : 1;
            for (int y = y0; y < y0 + ch; y++) {
                for (int x = x0; x < x0 + cw; x++) {
                    if (kind == 'n') {                   // 同 FramePreprocessorTest.goldenImage 的噪声：每个通道走一步 LCG
                        s = s * 6364136223846793005L + 1442695040888963407L;
                        int r = (int) (s >>> 56);
                        s = s * 6364136223846793005L + 1442695040888963407L;
                        int g = (int) (s >>> 56);
                        s = s * 6364136223846793005L + 1442695040888963407L;
                        int b = (int) (s >>> 56);
                        img[y * w + x] = 0xFF000000 | (r << 16) | (g << 8) | b;
                    } else if (kind == 'c') {
                        img[y * w + x] = color;
                    } else if ((x + y) % step == 0) {
                        img[y * w + x] = 0xFFFFFFFF;
                    }
                }
            }
        }
        return img;
    }

    @Test
    public void matchesPythonGolden() throws IOException {
        InputStream in = getClass().getClassLoader().getResourceAsStream("video/screen_crop_golden.txt");
        assertNotNull("缺少测试资源 video/screen_crop_golden.txt（python3 bench/tools/video_rescore.py golden 生成）", in);
        BufferedReader br = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
        ScreenCropper cropper = null;
        String name = null, line;
        int cases = 0, frames = 0;
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.startsWith("#") || line.isEmpty()) continue;
            String[] f = line.split("\\s+");
            if ("case".equals(f[0])) {
                name = f[1];
                cropper = new ScreenCropper();       // 每个 case 一个新的：它记着上一次裁的范围
                cases++;
                continue;
            }
            int w = Integer.parseInt(f[1]), h = Integer.parseInt(f[2]);
            int[] want = {Integer.parseInt(f[3]), Integer.parseInt(f[4]), Integer.parseInt(f[5]), Integer.parseInt(f[6])};
            int[] got = cropper.update(frame(w, h, f, 7), w, h);
            assertArrayEquals(name + " 第 " + frames + " 帧（" + line + "）", want, got);
            frames++;
        }
        assertTrue("对拍数据里的 case 太少", cases >= 10 && frames > cases);
    }

    @Test
    public void squareInsideVideoMatchesTraining() {
        // 2400×1080 的手机截一半：截图没跟上转屏时，16:9 的视频在竖屏截图里是 (54, 478, 432, 243)
        int w = 540, h = 1200;
        int[] img = frame(w, h, new String[]{"54,478,432,243,c33aa55"}, 0);
        int[] video = new ScreenCropper().update(img, w, h);
        assertArrayEquals(new int[]{54, 478, 432, 243}, video);
        // 再取居中正方形：就是视频自己的居中正方形（训练素材的取法），一点黑边都不带
        assertArrayEquals(new int[]{148, 478, 243, 243}, FramePreprocessor.cropRect(video));
        int[] out = FramePreprocessor.toModelInput(img, w, h, FramePreprocessor.cropRect(video));
        for (int p : out) assertTrue(String.format("%08x", p), p == 0xFF33AA55);
        // 不去黑边（2026-10-19 之前）：整个截图的居中正方形，上下大半是黑的
        int black = 0;
        for (int p : FramePreprocessor.toModelInput(img, w, h, FramePreprocessor.cropRect(w, h))) {
            if (!ScreenCropper.lit(p)) black++;
        }
        assertTrue("黑的像素 " + black, black > FramePreprocessor.MODEL_SIZE * FramePreprocessor.MODEL_SIZE / 2);
    }

    @Test
    public void wholeFrameWhenNothingToTrim() {
        int w = 1200, h = 540;
        int[] img = frame(w, h, new String[]{"0,0,1200,540,n7"}, 0);
        assertArrayEquals(new int[]{0, 0, w, h}, new ScreenCropper().update(img, w, h));
        assertTrue("默认去黑边", ScreenCropper.ENABLED);
    }
}
