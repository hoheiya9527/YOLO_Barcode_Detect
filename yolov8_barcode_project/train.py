#!/usr/bin/env python3
"""
YOLOv8 条形码检测训练脚本
支持自动设备检测（CUDA/ROCm/CPU）
"""

import argparse
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO


def check_device():
    """检测可用的计算设备"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"✓ 检测到GPU: {device_name}")
        return 'cuda'
    else:
        print("⚠ 未检测到CUDA设备，将使用CPU训练")
        print("提示：AMD显卡需要ROCm支持，或考虑使用Google Colab")
        return 'cpu'


def load_config(config_path):
    """加载训练配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except UnicodeDecodeError:
        with open(config_path, 'r', encoding='utf-8-sig') as f:
            config = yaml.safe_load(f)
    return config


def train(config_path, resume=False):
    """
    执行模型训练
    
    Args:
        config_path: 配置文件路径
        resume: 是否从上次中断处继续训练
    """
    config = load_config(config_path)
    
    # 检测设备
    if not config.get('device'):
        config['device'] = check_device()
    
    print(f"\n{'='*60}")
    print(f"训练配置:")
    print(f"  模型: {config['model']}")
    print(f"  数据集: {config['data']}")
    print(f"  训练轮数: {config['epochs']}")
    print(f"  批次大小: {config['batch']}")
    print(f"  图像尺寸: {config['imgsz']}")
    print(f"  设备: {config['device']}")
    print(f"{'='*60}\n")
    
    # 加载模型
    if resume:
        last_checkpoint = Path(config['project']) / config['name'] / 'weights' / 'last.pt'
        if last_checkpoint.exists():
            print(f"从检查点恢复训练: {last_checkpoint}")
            model = YOLO(str(last_checkpoint))
        else:
            raise FileNotFoundError(f"未找到检查点文件: {last_checkpoint}")
    else:
        model = YOLO(config['model'])
    
    # 开始训练
    results = model.train(
        data=config['data'],
        epochs=config['epochs'],
        batch=config['batch'],
        imgsz=config['imgsz'],
        device=config['device'],
        optimizer=config['optimizer'],
        lr0=config['lr0'],
        lrf=config['lrf'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay'],
        hsv_h=config['hsv_h'],
        hsv_s=config['hsv_s'],
        hsv_v=config['hsv_v'],
        degrees=config['degrees'],
        translate=config['translate'],
        scale=config['scale'],
        shear=config['shear'],
        perspective=config['perspective'],
        flipud=config['flipud'],
        fliplr=config['fliplr'],
        mosaic=config['mosaic'],
        mixup=config['mixup'],
        patience=config['patience'],
        save=config['save'],
        save_period=config['save_period'],
        workers=config['workers'],
        project=config['project'],
        name=config['name'],
        exist_ok=config['exist_ok'],
        val=config['val'],
        plots=config['plots']
    )
    
    print(f"\n{'='*60}")
    print("训练完成！")
    print(f"最佳模型: {Path(config['project']) / config['name'] / 'weights' / 'best.pt'}")
    print(f"最后模型: {Path(config['project']) / config['name'] / 'weights' / 'last.pt'}")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 条形码检测训练')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='训练配置文件路径'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='从上次中断处继续训练'
    )
    
    args = parser.parse_args()
    
    if not Path(args.config).exists():
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    
    train(args.config, args.resume)


if __name__ == '__main__':
    main()
