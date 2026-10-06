package com.example.helloworld;

import android.content.Context;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;


public class AudioInferenceHelper {

    private static final String TAG = "AudioInferenceHelper";

    private Interpreter yamnetInterpreter;
    private Interpreter classifierInterpreter;

    private static final int SAMPLE_RATE = 16000;
    private static final int INPUT_LENGTH = 32000; // 2秒音频数据
    private static final int EMBEDDING_SIZE = 1024;

    private NnApiDelegate nnApiDelegate;

    public AudioInferenceHelper(Context context) {
        try {
            nnApiDelegate = new NnApiDelegate();

            Interpreter.Options options = new Interpreter.Options();
            // 不使用 delegate，默认用 CPU，确保稳定跑通
            // options.addDelegate(new NnApiDelegate());

            // 加载两个 TFLite 模型（建议放到 assets 文件夹中）
            yamnetInterpreter = new Interpreter(loadModelFile(context, "yamnet.tflite"), options);
            classifierInterpreter = new Interpreter(loadModelFile(context, "yamnet_finetuned.tflite"), options);

            Log.d(TAG, "音频模型初始化成功");

        } catch (Exception e) {
            Log.e(TAG, "初始化失败: " + e.getMessage());
        }
    }

    // 加载 TFLite 模型文件
    private MappedByteBuffer loadModelFile(Context context, String modelFileName) throws IOException {
        FileInputStream inputStream = context.getAssets().openFd(modelFileName).createInputStream();
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = context.getAssets().openFd(modelFileName).getStartOffset();
        long declaredLength = context.getAssets().openFd(modelFileName).getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public class AudioInferenceResult {
        public final int index;
        public final float confidence;
        // 三分类完整概率 [sex, oral, noise]，供分组概率决策使用；无效结果时为空数组
        public final float[] probs;
        public AudioInferenceResult(int index, float confidence, float[] probs) {
            this.index = index;
            this.confidence = confidence;
            this.probs = probs != null ? probs : new float[0];
        }
    }


    /**
     * 对一个 2 秒音频窗口执行推理
     * @param audioBuffer  长度为 32000 的 float[] 音频数据
     * @return 返回最终预测类别索引
     */
    public AudioInferenceResult predict(float[] audioBuffer) {
        if (audioBuffer.length != INPUT_LENGTH) {
            Log.e(TAG, "音频输入长度不正确，期望: " + INPUT_LENGTH + "，实际: " + audioBuffer.length);
            return new AudioInferenceResult(-1, 0f, null);
        }
        Log.d(TAG, "AudioInferenceHelper: 开始推理 ");

        // 1. 输入 YAMNet 模型
        Object[] inputArray = { audioBuffer };

        // ⚠️ YAMNet 输出 shape 为 [num_frames, 1024]，不是 [1, 96, 1024]
        float[][] yamnetEmbeddingOutput = new float[4][EMBEDDING_SIZE]; // 实际推理出来是 [4, 1024]
        yamnetInterpreter.runForMultipleInputsOutputs(inputArray,
                java.util.Collections.singletonMap(1, yamnetEmbeddingOutput));

        // 2. 对 embedding 做平均
        int frameCount = yamnetEmbeddingOutput.length;
        float[] meanEmbedding = new float[EMBEDDING_SIZE];
        for (int i = 0; i < frameCount; i++) {
            for (int j = 0; j < EMBEDDING_SIZE; j++) {
                meanEmbedding[j] += yamnetEmbeddingOutput[i][j];
            }
        }
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
            meanEmbedding[j] /= frameCount;
        }

        // 3. 输入分类器模型
        float[][] classifierInput = new float[1][EMBEDDING_SIZE];
        classifierInput[0] = meanEmbedding;
        int outputClasses = classifierInterpreter.getOutputTensor(0).shape()[1];
        float[][] classifierOutput = new float[1][outputClasses];

        classifierInterpreter.run(classifierInput, classifierOutput);
        Log.d(TAG, "分类器输出: " + Arrays.toString(classifierOutput[0]));

        // 4. 获取最大类索引和置信度
        float[] scores = classifierOutput[0];
        int bestIndex = 0;
        float maxScore = scores[0];
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                bestIndex = i;
            }
        }
        return new AudioInferenceResult(bestIndex, maxScore, scores.clone());

    }


    public void close() {
        if (yamnetInterpreter != null) yamnetInterpreter.close();
        if (classifierInterpreter != null) classifierInterpreter.close();
        if (nnApiDelegate != null) nnApiDelegate.close();
    }
}
