#!/usr/bin/env python3
"""
快速开始脚本：一键完成训练、导出和推理
"""

import subprocess
import sys
import urllib.request
from pathlib import Path


def download_model(model_path):
    """下载YOLOv8n预训练模型"""
    if model_path.exists():
        print(f"✓ 模型文件已存在: {model_path}")
        return True
    
    print(f"\n{'='*60}")
    print("下载YOLOv8n预训练模型")
    print(f"{'='*60}\n")
    
    mirrors = [
        "https://gh.llkk.cc/https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "https://gh-proxy.top/https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
    ]
    
    for i, url in enumerate(mirrors, 1):
        try:
            print(f"尝试镜像 {i}/{len(mirrors)}: {url}")
            
            def progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                bar_length = 40
                filled = int(bar_length * percent / 100)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f'\r下载进度: [{bar}] {percent:.1f}%', end='', flush=True)
            
            urllib.request.urlretrieve(url, str(model_path), progress_hook)
            print(f"\n✓ 模型下载成功: {model_path}\n")
            return True
            
        except Exception as e:
            print(f"\n✗ 下载失败: {e}")
            if model_path.exists():
                model_path.unlink()
            continue
    
    print("\n✗ 所有镜像下载失败")
    print("请手动下载模型文件:")
    print("  URL: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt")
    print(f"  保存到: {model_path.absolute()}")
    return False


def run_command(cmd, description):
    """执行命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n✗ {description} 失败")
        sys.exit(1)
    
    print(f"\n✓ {description} 完成")


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║         YOLOv8 条形码检测 - 快速开始                      ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # 0. 下载预训练模型
    model_file = Path("yolov8n.pt")
    if not download_model(model_file):
        sys.exit(1)
    
    # 1. 训练模型
    print("\n步骤 1/4: 训练模型")
    print("提示：训练可能需要较长时间，可以随时按Ctrl+C中断")
    input("按Enter键开始训练...")
    
    run_command(
        "python train.py --config config.yaml",
        "模型训练"
    )
    
    # 2. 验证模型
    print("\n步骤 2/4: 验证模型")
    best_model = Path("runs/train/barcode_detection/weights/best.pt")
    
    if not best_model.exists():
        print(f"✗ 未找到训练好的模型: {best_model}")
        sys.exit(1)
    
    run_command(
        f"python validate.py --model {best_model}",
        "模型验证"
    )
    
    # 3. 导出TFLite
    print("\n步骤 3/4: 导出TFLite INT8模型（Android部署）")
    
    run_command(
        f"python export_tflite.py --model {best_model}",
        "TFLite导出"
    )
    
    # 4. 推理测试
    print("\n步骤 4/4: 推理测试（使用验证集）")
    test_images = Path("../test/images")
    
    if not test_images.exists():
        print(f"⚠ 测试图像目录不存在，跳过推理测试")
    else:
        run_command(
            f"python validate.py --model {best_model}",
            "验证集推理测试"
        )
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                   全部完成！                              ║
╠══════════════════════════════════════════════════════════╣
║  训练模型: {best_model}
║  TFLite模型: best_saved_model/best_int8.tflite
║  
║  Android集成说明: android/INTEGRATION_GUIDE.txt
╚══════════════════════════════════════════════════════════╝
    """)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ 用户中断")
        sys.exit(0)
