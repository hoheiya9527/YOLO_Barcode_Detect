#!/usr/bin/env python3
"""
YOLOv8 模型导出为TFLite INT8格式
针对Android移动端优化
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def export_to_tflite(
    model_path,
    output_dir=None,
    imgsz=640,
    int8=True
):
    """
    导出YOLOv8模型为TFLite格式
    
    Args:
        model_path: PyTorch模型路径 (.pt)
        output_dir: 输出目录，默认为模型所在目录
        imgsz: 输入图像尺寸
        int8: 是否使用INT8量化
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("TFLite导出配置:")
    print(f"  输入模型: {model_path}")
    print(f"  图像尺寸: {imgsz}")
    print(f"  INT8量化: {int8}")
    print(f"{'='*60}\n")
    
    # 加载模型
    model = YOLO(str(model_path))
    
    # 导出为TFLite
    print("开始导出TFLite模型...")
    export_path = model.export(
        format='tflite',
        imgsz=imgsz,
        int8=int8
    )
    
    # 移动到指定目录
    if output_dir:
        export_path_obj = Path(export_path)
        new_path = output_dir / export_path_obj.name
        export_path_obj.rename(new_path)
        export_path = new_path
    
    print(f"\n✓ TFLite模型导出成功: {export_path}")
    
    # 打印模型信息
    model_size = Path(export_path).stat().st_size / (1024 * 1024)
    print(f"\n模型信息:")
    print(f"  文件大小: {model_size:.2f} MB")
    print(f"  量化类型: {'INT8' if int8 else 'FP32'}")
    print(f"  输入尺寸: {imgsz}x{imgsz}")
    
    print(f"\n{'='*60}\n")
    
    return export_path


def main():
    parser = argparse.ArgumentParser(description='导出YOLOv8模型为TFLite格式')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='PyTorch模型路径 (.pt文件)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录，默认为模型所在目录'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='输入图像尺寸'
    )
    parser.add_argument(
        '--no-int8',
        action='store_true',
        help='不使用INT8量化（默认使用）'
    )
    
    args = parser.parse_args()
    
    export_to_tflite(
        model_path=args.model,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        int8=not args.no_int8
    )


if __name__ == '__main__':
    main()
