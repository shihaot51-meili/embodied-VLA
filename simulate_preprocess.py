#!/usr/bin/env python3
"""
阶段1：数据前处理（模拟）
============================================================
对应文档第4节：scripts/preprocess_lerobot_data_fast.py

模拟对原始 lerobot 示教数据进行前处理，核心做三件事：
  1. state 控制平滑：--action_to_next_state，将 action 赋值为下一帧 state
  2. 控制空间转换：--ee-mode（主板任务），将 state/action 从关节空间转 EE 空间
  3. 异常值滤除：--fix_outliers

本脚本用 numpy 生成 200~400 条模拟示教数据，模拟上述清洗过程，
输出处理后的 data/、videos/、meta/ 子目录结构。
"""

import json
import os
import numpy as np
import random
import shutil

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
NUM_EPISODES = random.randint(200, 400)        # 随机 200~400 条示教
FRAMES_PER_EPISODE = 100                        # 每条示教 100 帧
JOINT_DIM = 14                                  # 左臂7 + 右臂7
GRIPPER_DIM = 2                                 # 左爪 + 右爪
TORSO_DIM = 6                                   # 躯干6维
STATE_DIM = TORSO_DIM + JOINT_DIM + GRIPPER_DIM  # 6+14+2 = 22
ACTION_DIM = STATE_DIM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_raw_episode(episode_id, ee_mode=False):
    """生成一条原始示教数据（含异常值和不对齐问题）。"""
    rng = np.random.default_rng(episode_id)
    frames = []
    base_joints = rng.uniform(-1.0, 1.0, size=STATE_DIM)

    for t in range(FRAMES_PER_EPISODE):
        # 正常轨迹 + 缓慢漂移
        noise = rng.normal(0, 0.02, size=STATE_DIM)
        state = base_joints + 0.005 * t + noise

        # 注入异常值（约 2% 的帧）
        if rng.random() < 0.02:
            state[rng.integers(STATE_DIM)] = rng.uniform(50, 500)

        # 故制造 action 与 state 不对齐（原始 action = 当前 state + 大噪声）
        action = state + rng.normal(0, 0.1, size=ACTION_DIM)

        frames.append({"state": state, "action": action})

    return frames


def preprocess_episode(frames, ee_mode=False):
    """模拟前处理三步。"""
    clean = []
    for t, frame in enumerate(frames):
        state = frame["state"].copy()
        action = frame["action"].copy()

        # ① fix_outliers：将超范围值 clip 到合理范围
        state = np.clip(state, -3.0, 3.0)
        action = np.clip(action, -3.0, 3.0)

        # ② action_to_next_state：action = 下一帧 state（平滑一致）
        if t + 1 < len(frames):
            action = frames[t + 1]["state"].copy()
            action = np.clip(action, -3.0, 3.0)
        else:
            action = state.copy()

        # ③ ee_mode：将关节空间转为 EE 空间（模拟转换，维度不变但语义改变）
        if ee_mode:
            # 模拟 FK：关节角度 -> 末端位姿（简化为线性映射）
            state = state[TORSO_DIM:]  # 去掉躯干
            action = action[TORSO_DIM:]

        clean.append({"state": state.tolist(), "action": action.tolist()})

    return clean


def run(base_dir, output_dir, ee_mode=False, task_name="water_hose_insertion"):
    prefix = "ee_filted" if ee_mode else "filted"
    out_path = os.path.join(output_dir, f"lerobot_260710_{task_name}_v5_{prefix}_rby1_{NUM_EPISODES}s")

    print(f"\n{'='*60}")
    print(f"【阶段1·数据前处理】")
    print(f"  任务: {task_name}  |  模式: {'EE' if ee_mode else 'Joint'}")
    print(f"  原始数据路径: {base_dir}")
    print(f"  输出路径:     {out_path}")
    print(f"  示教条数:     {NUM_EPISODES}")
    print(f"{'='*60}")

    # 模拟生成 + 清洗
    for ep_id in range(NUM_EPISODES):
        raw = generate_raw_episode(ep_id, ee_mode)
        clean = preprocess_episode(raw, ee_mode)

    # 创建输出目录结构（模拟 lerobot 格式）
    for sub in ["data", "videos", "meta"]:
        os.makedirs(os.path.join(out_path, sub), exist_ok=True)

    # 写 meta info
    meta = {
        "total_episodes": NUM_EPISODES,
        "total_frames": NUM_EPISODES * FRAMES_PER_EPISODE,
        "frames_per_episode": FRAMES_PER_EPISODE,
        "state_dim": STATE_DIM - (TORSO_DIM if ee_mode else 0),
        "action_dim": ACTION_DIM - (TORSO_DIM if ee_mode else 0),
        "ee_mode": ee_mode,
        "task": task_name,
    }
    with open(os.path.join(out_path, "meta", "info.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 写一条示例处理后的数据
    sample_clean = preprocess_episode(generate_raw_episode(0, ee_mode), ee_mode)
    with open(os.path.join(out_path, "data", "episode_000000.json"), "w") as f:
        json.dump(sample_clean, f)

    print(f"\n✅ 前处理完成！")
    print(f"   输出目录: {out_path}")
    print(f"   子目录: data/ videos/ meta/ 均已生成")
    print(f"   总帧数: {NUM_EPISODES * FRAMES_PER_EPISODE}")
    print(f"\n📌 要点回顾：")
    print(f"   1. fix_outliers        → 异常值 clip 到 [-3, 3]")
    print(f"   2. action_to_next_state→ action=下一帧state，保证平滑")
    print(f"   3. ee_mode={'True' if ee_mode else 'False'}            → {'EE空间转换' if ee_mode else '保持关节空间'}")
    return out_path


if __name__ == "__main__":
    base = os.path.join(BASE_DIR, "mock_raw_data")
    os.makedirs(base, exist_ok=True)

    # 水管任务（joint 模式）
    out_water = run(base, os.path.join(BASE_DIR, "mock_processed"),
                    ee_mode=False, task_name="water_hose_insertion")

    # 主板任务（EE 模式）
    out_mainboard = run(base, os.path.join(BASE_DIR, "mock_processed"),
                        ee_mode=True, task_name="cable_connect_mainboard")

    # 保存路径供后续脚本使用
    with open(os.path.join(BASE_DIR, "pipeline_state.json"), "w") as f:
        json.dump({
            "water_hose_processed": out_water,
            "mainboard_processed": out_mainboard,
            "num_episodes": NUM_EPISODES,
        }, f, indent=2)

    print(f"\n✅ 所有前处理完成，状态已保存到 pipeline_state.json")
