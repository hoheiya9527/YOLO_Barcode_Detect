#!/usr/bin/env python3
"""
使用ONNX Runtime进行YOLOv8推理
包含完整的前后处理流程
"""

import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort


class YOLOv8ONNXInference:
    """YOLOv8 ONNX推理类"""
    
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        初始化推理器
        
        Args:
            model_path: ONNX模型路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS的IoU阈值
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 加载ONNX模型
        self.session = ort.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider']
        )
        
        # 获取模型输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        
        print(f"✓ ONNX模型加载成功")
        print(f"  输入尺寸: {self.input_width}x{self.input_height}")
        print(f"  置信度阈值: {self.conf_threshold}")
        print(f"  NMS IoU阈值: {self.iou_threshold}")
    
    def preprocess(self, image):
        """
        图像预处理
        
        Args:
            image: BGR格式的输入图像
            
        Returns:
            preprocessed: 预处理后的图像张量
            ratio: 缩放比例
            (pad_w, pad_h): 填充尺寸
        """
        img_height, img_width = image.shape[:2]
        
        # 计算缩放比例
        ratio = min(self.input_width / img_width, self.input_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        # 缩放图像
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # 创建填充图像
        pad_w = (self.input_width - new_width) // 2
        pad_h = (self.input_height - new_height) // 2
        
        padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h+new_height, pad_w:pad_w+new_width] = resized
        
        # 转换为RGB并归一化
        preprocessed = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        preprocessed = preprocessed.astype(np.float32) / 255.0
        
        # 转换为NCHW格式
        preprocessed = np.transpose(preprocessed, (2, 0, 1))
        preprocessed = np.expand_dims(preprocessed, axis=0)
        
        return preprocessed, ratio, (pad_w, pad_h)
    
    def postprocess(self, outputs, ratio, pad):
        """
        后处理：解析模型输出并应用NMS
        
        Args:
            outputs: 模型输出
            ratio: 缩放比例
            pad: 填充尺寸 (pad_w, pad_h)
            
        Returns:
            boxes: 检测框 [[x1, y1, x2, y2], ...]
            scores: 置信度分数
            class_ids: 类别ID
        """
        pad_w, pad_h = pad
        
        # YOLOv8输出格式: [batch, 84, 8400] -> [batch, 8400, 84]
        predictions = np.squeeze(outputs[0]).T
        
        # 提取边界框和分数
        boxes = predictions[:, :4]
        scores = predictions[:, 4:].max(axis=1)
        class_ids = predictions[:, 4:].argmax(axis=1)
        
        # 置信度过滤
        mask = scores > self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # 转换边界框格式：中心点+宽高 -> 左上角+右下角
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        
        # 坐标还原到原图
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_w) / ratio
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_h) / ratio
        
        # NMS
        indices = self.nms(boxes_xyxy, scores, self.iou_threshold)
        
        return boxes_xyxy[indices], scores[indices], class_ids[indices]
    
    @staticmethod
    def nms(boxes, scores, iou_threshold):
        """
        非极大值抑制
        
        Args:
            boxes: 边界框 [[x1, y1, x2, y2], ...]
            scores: 置信度分数
            iou_threshold: IoU阈值
            
        Returns:
            保留的索引列表
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def predict(self, image):
        """
        对单张图像进行推理
        
        Args:
            image: BGR格式的输入图像
            
        Returns:
            boxes: 检测框
            scores: 置信度
            class_ids: 类别ID
            inference_time: 推理耗时(ms)
        """
        # 预处理
        input_tensor, ratio, pad = self.preprocess(image)
        
        # 推理
        start_time = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        inference_time = (time.time() - start_time) * 1000
        
        # 后处理
        boxes, scores, class_ids = self.postprocess(outputs, ratio, pad)
        
        return boxes, scores, class_ids, inference_time
    
    def draw_detections(self, image, boxes, scores, class_ids, class_names):
        """
        在图像上绘制检测结果
        
        Args:
            image: 原始图像
            boxes: 检测框
            scores: 置信度
            class_ids: 类别ID
            class_names: 类别名称列表
            
        Returns:
            绘制后的图像
        """
        result_image = image.copy()
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            
            # 绘制边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{class_names[int(class_id)]}: {score:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result_image, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
            cv2.putText(result_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return result_image


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 ONNX推理')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='ONNX模型路径'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='输入图像或目录路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='runs/inference',
        help='输出目录'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='置信度阈值'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='NMS IoU阈值'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='显示结果'
    )
    
    args = parser.parse_args()
    
    # 初始化推理器
    inferencer = YOLOv8ONNXInference(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # 类别名称
    class_names = ['-Switch01']
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取输入图像列表
    source_path = Path(args.source)
    if source_path.is_file():
        image_paths = [source_path]
    elif source_path.is_dir():
        image_paths = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png'))
    else:
        raise ValueError(f"无效的输入路径: {args.source}")
    
    print(f"\n开始推理，共 {len(image_paths)} 张图像\n")
    
    total_time = 0
    for image_path in image_paths:
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"⚠ 无法读取图像: {image_path}")
            continue
        
        # 推理
        boxes, scores, class_ids, inference_time = inferencer.predict(image)
        total_time += inference_time
        
        # 绘制结果
        result_image = inferencer.draw_detections(image, boxes, scores, class_ids, class_names)
        
        # 保存结果
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), result_image)
        
        print(f"✓ {image_path.name}: {len(boxes)} 个检测, {inference_time:.1f}ms")
        
        # 显示结果
        if args.show:
            cv2.imshow('Detection Result', result_image)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
    
    if args.show:
        cv2.destroyAllWindows()
    
    avg_time = total_time / len(image_paths) if image_paths else 0
    print(f"\n推理完成！")
    print(f"  平均推理时间: {avg_time:.1f}ms")
    print(f"  结果保存至: {output_dir}")


if __name__ == '__main__':
    main()
