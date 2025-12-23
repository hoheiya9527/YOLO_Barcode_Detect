#!/usr/bin/env python3
"""
YOLOv8 模型导出为ONNX格式
针对ONNX Runtime优化
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import onnx


def export_to_onnx(
    model_path,
    output_path=None,
    imgsz=640,
    opset=12,
    simplify=True,
    dynamic=False
):
    """
    导出YOLOv8模型为ONNX格式
    
    Args:
        model_path: PyTorch模型路径 (.pt)
        output_path: ONNX输出路径，默认与模型同名
        imgsz: 输入图像尺寸
        opset: ONNX opset版本
        simplify: 是否简化ONNX模型
        dynamic: 是否使用动态batch size
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"\n{'='*60}")
    print("ONNX导出配置:")
    print(f"  输入模型: {model_path}")
    print(f"  图像尺寸: {imgsz}")
    print(f"  OPSET版本: {opset}")
    print(f"  简化模型: {simplify}")
    print(f"  动态batch: {dynamic}")
    print(f"{'='*60}\n")
    
    # 加载模型
    model = YOLO(str(model_path))
    
    # 导出为ONNX
    print("开始导出ONNX模型...")
    export_path = model.export(
        format='onnx',
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic
    )
    
    # 如果指定了输出路径，重命名文件
    if output_path:
        output_path = Path(output_path)
        Path(export_path).rename(output_path)
        export_path = output_path
    
    print(f"\n✓ ONNX模型导出成功: {export_path}")
    
    # 验证ONNX模型
    print("\n验证ONNX模型...")
    try:
        onnx_model = onnx.load(str(export_path))
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX模型验证通过")
        
        # 打印模型信息
        print(f"\n模型信息:")
        print(f"  IR版本: {onnx_model.ir_version}")
        print(f"  生产者: {onnx_model.producer_name}")
        print(f"  输入节点:")
        for input_node in onnx_model.graph.input:
            shape = [dim.dim_value for dim in input_node.type.tensor_type.shape.dim]
            print(f"    - {input_node.name}: {shape}")
        print(f"  输出节点:")
        for output_node in onnx_model.graph.output:
            shape = [dim.dim_value for dim in output_node.type.tensor_type.shape.dim]
            print(f"    - {output_node.name}: {shape}")
            
    except Exception as e:
        print(f"⚠ ONNX模型验证失败: {e}")
    
    print(f"\n{'='*60}\n")
    
    return export_path


def main():
    parser = argparse.ArgumentParser(description='导出YOLOv8模型为ONNX格式')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='PyTorch模型路径 (.pt文件)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='ONNX输出路径，默认与模型同名'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='输入图像尺寸'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=12,
        help='ONNX opset版本'
    )
    parser.add_argument(
        '--no-simplify',
        action='store_true',
        help='不简化ONNX模型'
    )
    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='使用动态batch size'
    )
    
    args = parser.parse_args()
    
    export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
        dynamic=args.dynamic
    )


if __name__ == '__main__':
    main()
