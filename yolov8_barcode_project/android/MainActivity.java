package com.example.barcode;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.os.Bundle;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * 条形码检测示例Activity
 * 使用CameraX实时检测
 */
public class MainActivity extends AppCompatActivity {
    
    private static final int REQUEST_CODE_PERMISSIONS = 10;
    private static final String[] REQUIRED_PERMISSIONS = {Manifest.permission.CAMERA};
    
    private PreviewView previewView;
    private ImageView resultImageView;
    private BarcodeDetector detector;
    private ExecutorService cameraExecutor;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        previewView = findViewById(R.id.previewView);
        resultImageView = findViewById(R.id.resultImageView);
        
        // 初始化检测器
        detector = new BarcodeDetector(
            this,
            "barcode_detection.tflite",
            0.25f,
            0.45f
        );
        
        cameraExecutor = Executors.newSingleThreadExecutor();
        
        // 请求相机权限
        if (allPermissionsGranted()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(
                this,
                REQUIRED_PERMISSIONS,
                REQUEST_CODE_PERMISSIONS
            );
        }
    }
    
    /**
     * 启动相机
     */
    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = 
            ProcessCameraProvider.getInstance(this);
        
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                
                // 预览
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());
                
                // 图像分析
                ImageAnalysis imageAnalyzer = new ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build();
                
                imageAnalyzer.setAnalyzer(cameraExecutor, new BarcodeAnalyzer());
                
                // 选择后置摄像头
                CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;
                
                // 绑定生命周期
                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageAnalyzer
                );
                
            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }
    
    /**
     * 图像分析器
     */
    private class BarcodeAnalyzer implements ImageAnalysis.Analyzer {
        
        @Override
        public void analyze(@NonNull ImageProxy imageProxy) {
            // 转换为Bitmap
            Bitmap bitmap = toBitmap(imageProxy);
            
            if (bitmap != null) {
                // 检测
                List<BarcodeDetector.Detection> detections = detector.detect(bitmap);
                
                // 绘制结果
                Bitmap resultBitmap = drawDetections(bitmap, detections);
                
                // 更新UI
                runOnUiThread(() -> resultImageView.setImageBitmap(resultBitmap));
            }
            
            imageProxy.close();
        }
    }
    
    /**
     * ImageProxy转Bitmap
     */
    private Bitmap toBitmap(ImageProxy imageProxy) {
        ImageProxy.PlaneProxy[] planes = imageProxy.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();
        
        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();
        
        byte[] nv21 = new byte[ySize + uSize + vSize];
        
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);
        
        YuvImage yuvImage = new YuvImage(
            nv21,
            ImageFormat.NV21,
            imageProxy.getWidth(),
            imageProxy.getHeight(),
            null
        );
        
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(
            new Rect(0, 0, imageProxy.getWidth(), imageProxy.getHeight()),
            100,
            out
        );
        
        byte[] imageBytes = out.toByteArray();
        return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }
    
    /**
     * 绘制检测结果
     */
    private Bitmap drawDetections(Bitmap bitmap, List<BarcodeDetector.Detection> detections) {
        Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);
        
        Paint boxPaint = new Paint();
        boxPaint.setColor(Color.GREEN);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(4f);
        
        Paint textPaint = new Paint();
        textPaint.setColor(Color.GREEN);
        textPaint.setTextSize(40f);
        textPaint.setStyle(Paint.Style.FILL);
        
        Paint bgPaint = new Paint();
        bgPaint.setColor(Color.argb(128, 0, 255, 0));
        bgPaint.setStyle(Paint.Style.FILL);
        
        for (BarcodeDetector.Detection detection : detections) {
            android.graphics.RectF bbox = detection.bbox;
            
            // 绘制边界框
            canvas.drawRect(bbox, boxPaint);
            
            // 绘制标签
            String label = String.format("Barcode: %.2f", detection.confidence);
            Rect textBounds = new Rect();
            textPaint.getTextBounds(label, 0, label.length(), textBounds);
            
            canvas.drawRect(
                bbox.left,
                bbox.top - textBounds.height() - 10,
                bbox.left + textBounds.width() + 10,
                bbox.top,
                bgPaint
            );
            
            canvas.drawText(label, bbox.left + 5, bbox.top - 5, textPaint);
        }
        
        return mutableBitmap;
    }
    
    /**
     * 检查权限
     */
    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) 
                != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }
    
    @Override
    public void onRequestPermissionsResult(
        int requestCode,
        @NonNull String[] permissions,
        @NonNull int[] grantResults
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera();
            } else {
                finish();
            }
        }
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
        detector.close();
    }
}
