package com.example.barcode;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * YOLOv8 条形码检测器
 * 使用TFLite INT8量化模型
 */
public class BarcodeDetector {
    
    private static final int INPUT_SIZE = 640;
    private static final int NUM_CLASSES = 1;
    private static final int OUTPUT_SIZE = 8400;
    
    private Interpreter interpreter;
    private final float confThreshold;
    private final float iouThreshold;
    
    /**
     * 检测结果类
     */
    public static class Detection {
        public RectF bbox;
        public float confidence;
        public int classId;
        
        public Detection(RectF bbox, float confidence, int classId) {
            this.bbox = bbox;
            this.confidence = confidence;
            this.classId = classId;
        }
    }
    
    /**
     * 构造函数
     */
    public BarcodeDetector(Context context, String modelPath, float confThreshold, float iouThreshold) {
        this.confThreshold = confThreshold;
        this.iouThreshold = iouThreshold;
        loadModel(context, modelPath);
    }
    
    /**
     * 加载模型
     */
    private void loadModel(Context context, String modelPath) {
        try {
            Interpreter.Options options = new Interpreter.Options();
            // 使用NNAPI硬件加速
            options.setUseNNAPI(true);
            // 设置线程数
            options.setNumThreads(4);
            
            MappedByteBuffer modelBuffer = FileUtil.loadMappedFile(context, modelPath);
            interpreter = new Interpreter(modelBuffer, options);
            
            System.out.println("✓ TFLite模型加载成功");
        } catch (IOException e) {
            throw new RuntimeException("模型加载失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 检测图像中的条形码
     */
    public List<Detection> detect(Bitmap bitmap) {
        if (interpreter == null) {
            throw new IllegalStateException("模型未初始化");
        }
        
        // 预处理
        PreprocessResult preprocessResult = preprocess(bitmap);
        
        // 推理
        float[][][] outputArray = new float[1][84][OUTPUT_SIZE];
        interpreter.run(preprocessResult.inputBuffer, outputArray);
        
        // 后处理
        return postprocess(outputArray[0], preprocessResult.ratio, 
                          preprocessResult.padW, preprocessResult.padH);
    }
    
    /**
     * 预处理结果类
     */
    private static class PreprocessResult {
        ByteBuffer inputBuffer;
        float ratio;
        int padW;
        int padH;
        
        PreprocessResult(ByteBuffer inputBuffer, float ratio, int padW, int padH) {
            this.inputBuffer = inputBuffer;
            this.ratio = ratio;
            this.padW = padW;
            this.padH = padH;
        }
    }
    
    /**
     * 图像预处理
     */
    private PreprocessResult preprocess(Bitmap bitmap) {
        int originalWidth = bitmap.getWidth();
        int originalHeight = bitmap.getHeight();
        
        // 计算缩放比例
        float ratio = Math.min((float) INPUT_SIZE / originalWidth, 
                              (float) INPUT_SIZE / originalHeight);
        int newWidth = (int) (originalWidth * ratio);
        int newHeight = (int) (originalHeight * ratio);
        
        // 计算填充
        int padW = (INPUT_SIZE - newWidth) / 2;
        int padH = (INPUT_SIZE - newHeight) / 2;
        
        // 缩放图像
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true);
        
        // 创建填充后的图像
        Bitmap paddedBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        android.graphics.Canvas canvas = new android.graphics.Canvas(paddedBitmap);
        canvas.drawColor(android.graphics.Color.rgb(114, 114, 114));
        canvas.drawBitmap(resizedBitmap, padW, padH, null);
        
        // 转换为ByteBuffer
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3);
        inputBuffer.order(ByteOrder.nativeOrder());
        
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        paddedBitmap.getPixels(intValues, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        
        for (int pixelValue : intValues) {
            // RGB归一化到[0, 1]
            inputBuffer.putFloat(((pixelValue >> 16) & 0xFF) / 255.0f);
            inputBuffer.putFloat(((pixelValue >> 8) & 0xFF) / 255.0f);
            inputBuffer.putFloat((pixelValue & 0xFF) / 255.0f);
        }
        
        return new PreprocessResult(inputBuffer, ratio, padW, padH);
    }
    
    /**
     * 后处理：解析输出并应用NMS
     */
    private List<Detection> postprocess(float[][] output, float ratio, int padW, int padH) {
        List<Detection> detections = new ArrayList<>();
        
        // 解析输出: [84, 8400]
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            // 读取边界框 (cx, cy, w, h)
            float cx = output[0][i];
            float cy = output[1][i];
            float w = output[2][i];
            float h = output[3][i];
            
            // 读取类别分数
            float maxScore = 0f;
            int maxClassId = 0;
            for (int c = 0; c < NUM_CLASSES; c++) {
                float score = output[4 + c][i];
                if (score > maxScore) {
                    maxScore = score;
                    maxClassId = c;
                }
            }
            
            // 置信度过滤
            if (maxScore < confThreshold) continue;
            
            // 转换边界框格式并还原到原图坐标
            float x1 = ((cx - w / 2) - padW) / ratio;
            float y1 = ((cy - h / 2) - padH) / ratio;
            float x2 = ((cx + w / 2) - padW) / ratio;
            float y2 = ((cy + h / 2) - padH) / ratio;
            
            detections.add(new Detection(
                new RectF(x1, y1, x2, y2),
                maxScore,
                maxClassId
            ));
        }
        
        // NMS
        return nms(detections, iouThreshold);
    }
    
    /**
     * 非极大值抑制
     */
    private List<Detection> nms(List<Detection> detections, float iouThreshold) {
        // 按置信度降序排序
        Collections.sort(detections, (a, b) -> Float.compare(b.confidence, a.confidence));
        
        List<Detection> keep = new ArrayList<>();
        
        for (Detection detection : detections) {
            boolean shouldKeep = true;
            
            for (Detection kept : keep) {
                float iou = calculateIoU(detection.bbox, kept.bbox);
                if (iou > iouThreshold) {
                    shouldKeep = false;
                    break;
                }
            }
            
            if (shouldKeep) {
                keep.add(detection);
            }
        }
        
        return keep;
    }
    
    /**
     * 计算IoU
     */
    private float calculateIoU(RectF box1, RectF box2) {
        float intersectLeft = Math.max(box1.left, box2.left);
        float intersectTop = Math.max(box1.top, box2.top);
        float intersectRight = Math.min(box1.right, box2.right);
        float intersectBottom = Math.min(box1.bottom, box2.bottom);
        
        float intersectWidth = Math.max(0, intersectRight - intersectLeft);
        float intersectHeight = Math.max(0, intersectBottom - intersectTop);
        float intersectArea = intersectWidth * intersectHeight;
        
        float box1Area = box1.width() * box1.height();
        float box2Area = box2.width() * box2.height();
        float unionArea = box1Area + box2Area - intersectArea;
        
        return unionArea > 0 ? intersectArea / unionArea : 0;
    }
    
    /**
     * 释放资源
     */
    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
    }
}
