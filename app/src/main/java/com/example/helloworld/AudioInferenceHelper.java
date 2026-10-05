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
import java.util.HashMap;
import java.util.Map;


public class AudioInferenceHelper {

    private static final String TAG = "AudioInferenceHelper";

    private Interpreter yamnetInterpreter;
    private Interpreter classifierInterpreter;

    private static final int SAMPLE_RATE = 16000;
    private static final int INPUT_LENGTH = 32000; // 2秒音频数据
    private static final int EMBEDDING_SIZE = 1024;
    private static final int AUDIOSET_CLASSES = 521;
    // 32000 点输入 → 4 帧（每帧 0.96 秒、步长 0.48 秒，末帧补零）
    private static final int YAMNET_FRAMES = 4;

    /**
     * 写进评测 CSV（a_ys_* 列）的 AudioSet 类别：YAMNet output 0 的列号，按帧取平均。
     * 前 4 类是"剧情"候选证据（对白、旁白、配乐），其余是"动作"候选。
     * 与 BenchLogger 表头、bench/tools/metrics.py 的 YS_CLASSES 一一对应（metrics.py --selftest 核对）。
     */
    public static final int[] AUDIOSET_LOG_CLASSES = {0, 2, 3, 132, 494, 22, 33, 36, 40, 461};
    public static final String[] AUDIOSET_LOG_NAMES = {"speech", "conversation", "narration", "music",
            "silence", "moan", "groan", "breathing", "pant", "slap"};

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
        // AUDIOSET_LOG_CLASSES 各类的 AudioSet 分数（各帧平均），只写评测 CSV；无效结果时为空数组
        public final float[] audioSet;
        public AudioInferenceResult(int index, float confidence, float[] probs) {
            this(index, confidence, probs, null);
        }
        public AudioInferenceResult(int index, float confidence, float[] probs, float[] audioSet) {
            this.index = index;
            this.confidence = confidence;
            this.probs = probs != null ? probs : new float[0];
            this.audioSet = audioSet != null ? audioSet : new float[0];
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

        // YAMNet 有 3 个输出：0 = AudioSet 521 类分数 [帧, 521]，1 = embedding [帧, 1024]，2 = log-mel。
        // 整张图每次都会算完，多取 output 0 只是多拷 4x521 个 float。
        // ⚠️ YAMNet 输出 shape 为 [num_frames, 1024]，不是 [1, 96, 1024]
        float[][] yamnetScores = new float[YAMNET_FRAMES][AUDIOSET_CLASSES];
        float[][] yamnetEmbeddingOutput = new float[YAMNET_FRAMES][EMBEDDING_SIZE]; // 实际推理出来是 [4, 1024]
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, yamnetScores);
        outputs.put(1, yamnetEmbeddingOutput);
        yamnetInterpreter.runForMultipleInputsOutputs(inputArray, outputs);

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

        // 5. 挑出的 AudioSet 类别分数，各帧平均（bench/tools/audio_rescore.py 离线重算时同样按帧平均）
        float[] audioSet = new float[AUDIOSET_LOG_CLASSES.length];
        for (int k = 0; k < AUDIOSET_LOG_CLASSES.length; k++) {
            float s = 0f;
            for (int i = 0; i < frameCount; i++) {
                s += yamnetScores[i][AUDIOSET_LOG_CLASSES[k]];
            }
            audioSet[k] = s / frameCount;
        }
        return new AudioInferenceResult(bestIndex, maxScore, scores.clone(), audioSet);

    }


    public void close() {
        if (yamnetInterpreter != null) yamnetInterpreter.close();
        if (classifierInterpreter != null) classifierInterpreter.close();
        if (nnApiDelegate != null) nnApiDelegate.close();
    }
}
