#!/usr/bin/env python3
"""
pi05 微调全流程模拟（主控脚本）
============================================================
依次执行五个阶段：
  阶段1: 数据前处理      → simulate_preprocess.py
  阶段2: 合并数据集      → simulate_merge.py
  阶段3: 编写训练 config → simulate_config.py
  阶段4: 计算 norm       → simulate_norm.py
  阶段5: 启动训练        → simulate_train.py

用法:
    python run_all.py
"""

import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STAGES = [
    ("阶段1·数据前处理",      "simulate_preprocess.py"),
    ("阶段2·合并数据集",      "simulate_merge.py"),
    ("阶段3·编写训练config",   "simulate_config.py"),
    ("阶段4·计算归一化norm",   "simulate_norm.py"),
    ("阶段5·启动训练",        "simulate_train.py"),
]


def run_stage(label, script_name):
    print(f"\n{'#'*60}")
    print(f"##  开始 {label}")
    print(f"##  脚本: {script_name}")
    print(f"{'#'*60}")

    script_path = os.path.join(BASE_DIR, script_name)
    result = subprocess.run([sys.executable, script_path], cwd=BASE_DIR)

    if result.returncode != 0:
        print(f"\n❌ {label} 执行失败 (exit code {result.returncode})")
        sys.exit(1)

    print(f"\n✅ {label} 执行完成")


if __name__ == "__main__":
    print("=" * 60)
    print("  pi05 微调全流程模拟")
    print("  模拟 200~400 条示教数据的完整训练流水线")
    print("  任务: 水管插入(joint+RTC) + 主板插线(EE)")
    print("=" * 60)

    for label, script in STAGES:
        run_stage(label, script)

    print("\n" + "=" * 60)
    print("  🎉 全部 5 个阶段模拟完成！")
    print("=" * 60)
    print("\n生成的关键产物:")
    print("  - mock_processed/        前处理后的数据集")
    print("  - mock_merged/            合并后的数据集")
    print("  - training_configs.json   两个任务的训练配置")
    print("  - norm_stats.json         归一化统计量（各数据集目录下）")
    print("  - mock_checkpoints/       模拟训练产出的 checkpoint + loss.log")
    print("  - pipeline_state.json     流水线状态（各阶段路径）")
