#!/usr/bin/env python3
"""
阶段4：计算归一化统计量 norm（模拟）
============================================================
对应文档第7节：scripts/compute_norm_states_fast.py

根据 config 名定位数据集，统计 state 和 action 的均值/方差/中位数，
写出 norm_stats.json，位置即 config 中 assets 所指。

关键分支（最易出错！）：
  - joint 模式（水管）  → 必须加 --torso 6（跳过前6维躯干关节）
  - EE 模式（主板）    → 禁止加 --torso 6
  - 夹爪越界            → 追加 --correct 修正数据（直接改文件）
"""

import json
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_norm_stats(dataset_path, config_name, torso=0, correct=False):
    """
    模拟计算 norm_stats.json。
    torso > 0 时跳过前 torso 维（joint 模式）。
    correct=True 时修正夹爪越界值。
    """
    print(f"\n{'='*60}")
    print(f"【阶段4·计算归一化统计量】")
    print(f"{'='*60}")
    print(f"  config:  {config_name}")
    print(f"  dataset: {dataset_path}")
    print(f"  torso:   {torso if torso > 0 else '不跳过（EE模式）'}")
    print(f"  correct: {correct}")

    # 读取数据集 meta
    meta_file = os.path.join(dataset_path, "meta", "info.json")
    with open(meta_file) as f:
        meta = json.load(f)

    total_frames = meta["total_frames"]
    state_dim_raw = meta["state_dim"]

    # 模拟生成统计数据（实际脚本遍历所有帧计算）
    rng = np.random.default_rng(42)
    # 模拟 state/action 分布
    effective_dim = state_dim_raw - (torso if torso > 0 else 0)

    state_mean = rng.uniform(-0.5, 0.5, effective_dim)
    state_std  = rng.uniform(0.05, 0.3, effective_dim)
    state_min  = state_mean - 3 * state_std
    state_max  = state_mean + 3 * state_std
    state_median = state_mean.copy()

    action_mean = state_mean.copy()
    action_std  = state_std.copy()
    action_min  = state_min.copy()
    action_max  = state_max.copy()
    action_median = action_mean.copy()

    if correct:
        print(f"  --correct: 检测到夹爪越界值，已修正...")
        # 模拟修正：将夹爪值 clip 到 [0, 0.1]
        gripper_start = effective_dim - 2
        state_mean[gripper_start:] = np.clip(state_mean[gripper_start:], 0, 0.1)

    # 构建 norm_stats.json 结构（与真实格式一致）
    norm_stats = {
        "state": {
            "mean": state_mean.tolist(),
            "std": state_std.tolist(),
            "min": state_min.tolist(),
            "max": state_max.tolist(),
            "median": state_median.tolist(),
        },
        "action": {
            "mean": action_mean.tolist(),
            "std": action_std.tolist(),
            "min": action_min.tolist(),
            "max": action_max.tolist(),
            "median": action_median.tolist(),
        },
        "num_frames": total_frames,
        "torso_skipped": torso,
        "config_name": config_name,
    }

    # 确定输出路径：assets_dir / asset_id / norm_stats.json
    assets_dir = os.path.dirname(dataset_path)
    asset_id = os.path.basename(dataset_path)
    norm_dir = os.path.join(assets_dir, asset_id)
    os.makedirs(norm_dir, exist_ok=True)
    norm_path = os.path.join(norm_dir, "norm_stats.json")

    with open(norm_path, "w") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"\n  ✅ norm_stats.json 已生成: {norm_path}")
    print(f"     state 维度: {effective_dim} (原始 {state_dim_raw}, 跳过躯干 {torso})")
    print(f"     总帧数: {total_frames}")
    print(f"     state mean[0:3]: {[f'{x:.4f}' for x in state_mean[:3]]}")
    print(f"     state std[0:3]:  {[f'{x:.4f}' for x in state_std[:3]]}")

    return norm_path


if __name__ == "__main__":
    state_file = os.path.join(BASE_DIR, "pipeline_state.json")
    with open(state_file) as f:
        state = json.load(f)

    with open(os.path.join(BASE_DIR, "training_configs.json")) as f:
        configs = json.load(f)

    # ── 水管任务（joint 模式）→ 必须加 --torso 6 ──
    water_norm = compute_norm_stats(
        dataset_path=state["water_hose_merged"],
        config_name=configs["water_hose"]["name"],
        torso=6,       # ← joint 模式必须加！
        correct=False,
    )

    # ── 主板任务（EE 模式）→ 禁止加 --torso ──
    mb_norm = compute_norm_stats(
        dataset_path=state["mainboard_merged"],
        config_name=configs["mainboard"]["name"],
        torso=0,       # ← EE 模式不加！
        correct=False,
    )

    state["water_norm_path"] = water_norm
    state["mainboard_norm_path"] = mb_norm
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

    print(f"\n📌 要点回顾：")
    print(f"   1. joint 模式（水管）→ 必须加 --torso 6，跳过前6维躯干")
    print(f"   2. EE 模式（主板）  → 禁止加 --torso 6")
    print(f"   3. 夹爪越界（0~0.1外）→ 加 --correct 修正（直接改数据文件）")
    print(f"   4. norm_stats.json 存于 assets_dir/asset_id/ 下，训练与推理共用同一份")
    print(f"   5. 复用旧 norm 可跳过本步，只需 assets 指向正确位置")
    print(f"\n⚠️  此处是最易出错的分支，加错或漏加都会导致模型训坏！")
