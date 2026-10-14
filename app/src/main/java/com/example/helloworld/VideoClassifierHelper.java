package com.example.helloworld;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

/**
 * VideoClassifierHelper
 *
 * MobileNetV3Small + Temporal Average Pooling 三分类视频判别器。
 *
 * 输入：最近 3 秒内、间隔 250ms 采样的 12 帧 RGB 图像（每帧 160x160）。
 * 输出：[normal_plot, oral, sex] 三类 softmax 概率。
 *   - index 0 -> normal_plot（剧情/非动作）
 *   - index 1 -> oral
 *   - index 2 -> sex
 *
 * 注意：模型内部包含 Rescaling 层（include_preprocessing=True），
 * 会自动把 0-255 归一化到 -1~1。因此这里输入的是原始像素值，绝不做 /255 或标准化。
 */
public class VideoClassifierHelper {

    private static final String TAG = "VideoClassifierHelper";

    private static final String MODEL_FILE = "mobilenetv3small_tap_3class_float32.tflite";

    /** 单次推理消费的帧数（3 秒 / 250ms）。 */
    public static final int NUM_FRAMES = 12;
    /** 模型输入的高/宽（预处理见 FramePreprocessor）。 */
    public static final int INPUT_SIZE = FramePreprocessor.MODEL_SIZE;

    private static final int CHANNELS = 3;
    private static final int NUM_CLASSES = 3;

    private Interpreter interpreter;

    // 复用的输入缓冲区：1 * 12 * 160 * 160 * 3 * 4 = 3,686,400 字节
    private final ByteBuffer inputBuffer;
    // 复用的单帧像素缓冲区
    private final int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];

    public VideoClassifierHelper(Context context) {
        inputBuffer = ByteBuffer.allocateDirect(
                NUM_FRAMES * INPUT_SIZE * INPUT_SIZE * CHANNELS * 4);
        inputBuffer.order(ByteOrder.nativeOrder());

        try {
            Interpreter.Options options = new Interpreter.Options();
            // 模型仅 4MB，CPU 单次推理 30-60ms，远小于 1000ms 步长，CPU 多线程即可稳定跑通
            options.setNumThreads(4);
            interpreter = new Interpreter(loadModelFile(context, MODEL_FILE), options);
            Log.d(TAG, "视频分类模型初始化成功: " + MODEL_FILE
                    + ", 输入 shape=" + Arrays.toString(interpreter.getInputTensor(0).shape())
                    + ", 输出 shape=" + Arrays.toString(interpreter.getOutputTensor(0).shape()));
        } catch (Exception e) {
            Log.e(TAG, "视频分类模型初始化失败: " + e.getMessage(), e);
        }
    }

    private MappedByteBuffer loadModelFile(Context context, String modelFileName) throws IOException {
        FileInputStream inputStream = context.getAssets().openFd(modelFileName).createInputStream();
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = context.getAssets().openFd(modelFileName).getStartOffset();
        long declaredLength = context.getAssets().openFd(modelFileName).getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /** 三分类结果。index：0=normal_plot, 1=oral, 2=sex；index<0 表示无效。 */
    public static class Result {
        public final int index;
        public final float confidence;
        public final float[] probs;

        Result(int index, float confidence, float[] probs) {
            this.index = index;
            this.confidence = confidence;
            this.probs = probs;
        }
    }

    /**
     * 对 12 帧执行一次推理。
     *
     * @param frames 长度为 {@link #NUM_FRAMES} 的 Bitmap 数组（按时间从旧到新）。
     * @return 三分类结果；模型未就绪或输入非法时返回 index=-1。
     */
    public Result predict(Bitmap[] frames) {
        if (interpreter == null) {
            Log.e(TAG, "interpreter 未初始化");
            return new Result(-1, 0f, new float[NUM_CLASSES]);
        }
        if (frames == null || frames.length != NUM_FRAMES) {
            Log.e(TAG, "输入帧数不正确，期望 " + NUM_FRAMES + "，实际 "
                    + (frames == null ? 0 : frames.length));
            return new Result(-1, 0f, new float[NUM_CLASSES]);
        }

        long tPrepStart = System.currentTimeMillis();

        // Bitmap -> Tensor：内存按 NHWC 排列，index = t*H*W*3 + h*W*3 + w*3 + c
        inputBuffer.rewind();
        for (int t = 0; t < NUM_FRAMES; t++) {
            Bitmap frame = frames[t];
            if (frame == null || frame.isRecycled()) {
                Log.e(TAG, "第 " + t + " 帧无效");
                return new Result(-1, 0f, new float[NUM_CLASSES]);
            }
            // 三种模式送进来的都应该已经是 160x160（toModelBitmap）；兜底也按训练的方式裁剪缩放，不再整帧压扁
            if (frame.getWidth() != INPUT_SIZE || frame.getHeight() != INPUT_SIZE) {
                frame = toModelBitmap(frame, null);
            }
            frame.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
            for (int p : pixels) {
                // 原始像素值 0-255，按 R, G, B 顺序，不做任何归一化
                inputBuffer.putFloat((p >> 16) & 0xFF); // R
                inputBuffer.putFloat((p >> 8) & 0xFF);  // G
                inputBuffer.putFloat(p & 0xFF);         // B
            }
        }

        long tPrepEnd = System.currentTimeMillis();

        float[][] output = new float[1][NUM_CLASSES];
        inputBuffer.rewind();
        try {
            interpreter.run(inputBuffer, output);
        } catch (Exception e) {
            Log.e(TAG, "推理异常: " + e.getMessage(), e);
            return new Result(-1, 0f, new float[NUM_CLASSES]);
        }

        long tInferEnd = System.currentTimeMillis();

        float[] probs = output[0];
        int bestIndex = 0;
        float bestProb = probs[0];
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > bestProb) {
                bestProb = probs[i];
                bestIndex = i;
            }
        }

        Log.d(TAG, String.format("预处理 %dms, 推理 %dms, 概率=%s",
                tPrepEnd - tPrepStart, tInferEnd - tPrepEnd, Arrays.toString(probs)));

        return new Result(bestIndex, bestProb, probs);
    }

    /**
     * 任意尺寸的画面 → 模型输入（160×160）：按 {@link FramePreprocessor} 居中裁正方形、面积平均缩小，与训练一致。
     * 网页模式（TextureView 截的帧）和在线模式（截屏）用它；离线模式在 VideoFrameExtractor 里已经处理好了。
     *
     * @param scratch 可复用的像素缓冲（长度不够或为 null 时新建），避免每帧分配几 MB
     */
    public static Bitmap toModelBitmap(Bitmap src, int[] scratch) {
        int w = src.getWidth(), h = src.getHeight();
        int[] px = (scratch != null && scratch.length >= w * h) ? scratch : new int[w * h];
        src.getPixels(px, 0, w, 0, 0, w, h);
        int[] out = FramePreprocessor.toModelInput(px, w, h, FramePreprocessor.cropRect(w, h));
        return Bitmap.createBitmap(out, INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
    }

    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
    }
}
