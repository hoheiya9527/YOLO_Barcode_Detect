#!/usr/bin/env python3
"""
YOLOv8模型验证脚本
评估模型在验证集上的性能
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def validate_model(model_path, data_path, imgsz=640, batch=16, device=''):
    """
    验证模型性能
    
    Args:
        model_path: 模型路径
        data_path: 数据集配置文件
        imgsz: 图像尺寸
        batch: 批次大小
        device: 计算设备
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"\n{'='*60}")
    print("验证配置:")
    print(f"  模型: {model_path}")
    print(f"  数据集: {data_path}")
    print(f"  图像尺寸: {imgsz}")
    print(f"  批次大小: {batch}")
    print(f"{'='*60}\n")
    
    # 加载模型
    model = YOLO(str(model_path))
    
    # 验证
    results = model.val(
        data=data_path,
        imgsz=imgsz,
        batch=batch,
        device=device,
        plots=True
    )
    
    # 打印结果
    print(f"\n{'='*60}")
    print("验证结果:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='YOLOv8模型验证')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='模型路径 (.pt文件)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='../data.yaml',
        help='数据集配置文件'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='图像尺寸'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='批次大小'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='计算设备 (cuda/cpu，留空自动选择)'
    )
    
    args = parser.parse_args()
    
    validate_model(
        model_path=args.model,
        data_path=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )


if __name__ == '__main__':
    main()
