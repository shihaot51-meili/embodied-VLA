#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pi05 微调全流程模拟（单文件版，纯标准库，无第三方依赖）
============================================================
模拟 200~400 条示教数据，完整走通五个阶段：
  1. 数据前处理
  2. 数据合并
  3. 编写训练 config
  4. 计算 norm
  5. 启动训练

运行: python pi05_pipeline_sim.py
"""

import json
import os
import math
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_LINES = []

def log(msg=""):
    LOG_LINES.append(str(msg))

# ════════════════════════════════════════════════════════════
#  全局配置
# ════════════════════════════════════════════════════════════
NUM_EPISODES = random.randint(200, 400)
FRAMES_PER_EPISODE = 100
JOINT_DIM = 14
GRIPPER_DIM = 2
TORSO_DIM = 6
STATE_DIM = TORSO_DIM + JOINT_DIM + GRIPPER_DIM  # 22


# ════════════════════════════════════════════════════════════
#  阶段1：数据前处理
# ════════════════════════════════════════════════════════════
def stage1_preprocess():
    log("\n" + "=" * 60)
    log("【阶段1·数据前处理】")
    log("  对应脚本: scripts/preprocess_lerobot_data_fast.py")
    log("=" * 60)
    log(f"  随机生成示教条数: {NUM_EPISODES}")
    log(f"  每条帧数: {FRAMES_PER_EPISODE}")

    results = {}

    for task_name, ee_mode in [("water_hose_insertion", False),
                                ("cable_connect_mainboard", True)]:
        prefix = "ee_filted" if ee_mode else "filted"
        out_name = f"lerobot_260710_{task_name}_v5_{prefix}_rby1_{NUM_EPISODES}s"
        out_path = os.path.join(BASE_DIR, "mock_processed", out_name)

        log(f"\n  --- 任务: {task_name} | 模式: {'EE' if ee_mode else 'Joint'} ---")
        log(f"  输出路径: {out_path}")

        # 模拟生成原始数据 + 清洗
        total_frames = 0
        for ep_id in range(NUM_EPISODES):
            for t in range(FRAMES_PER_EPISODE):
                # 模拟原始 state（含 ~2% 异常值）
                state = [random.gauss(0, 0.5) for _ in range(STATE_DIM)]
                if random.random() < 0.02:
                    idx = random.randint(0, STATE_DIM - 1)
                    state[idx] = random.uniform(50, 500)

                # ① fix_outliers: clip
                state = [max(-3.0, min(3.0, v)) for v in state]

                # ② action_to_next_state: action = 下一帧 state
                action = state  # 简化

                # ③ ee_mode: 去掉躯干6维
                if ee_mode:
                    state = state[TORSO_DIM:]
                    action = action[TORSO_DIM:]

                total_frames += 1

        # 创建目录结构
        for sub in ["data", "videos", "meta"]:
            os.makedirs(os.path.join(out_path, sub), exist_ok=True)

        # 写 meta
        eff_dim = STATE_DIM - (TORSO_DIM if ee_mode else 0)
        meta = {
            "total_episodes": NUM_EPISODES,
            "total_frames": total_frames,
            "frames_per_episode": FRAMES_PER_EPISODE,
            "state_dim": eff_dim,
            "action_dim": eff_dim,
            "ee_mode": ee_mode,
            "task": task_name,
        }
        with open(os.path.join(out_path, "meta", "info.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # 写示例 episode
        sample = []
        for t in range(FRAMES_PER_EPISODE):
            s = [round(random.gauss(0, 0.3), 4) for _ in range(eff_dim)]
            sample.append({"state": s, "action": s})
        with open(os.path.join(out_path, "data", "episode_000000.json"), "w", encoding="utf-8") as f:
            json.dump(sample, f)

        log(f"  ✅ 完成! 总帧数={total_frames}, state_dim={eff_dim}")
        log(f"     data/ videos/ meta/ 子目录均已生成")

        results[task_name] = out_path

    log(f"\n  📌 阶段1要点:")
    log(f"     1. fix_outliers        → 异常值 clip 到 [-3, 3]")
    log(f"     2. action_to_next_state→ action=下一帧state，保证平滑")
    log(f"     3. ee_mode             → EE空间转换(主板任务)，否则保持关节空间")
    log(f"     4. --output_dir 必填，否则原地修改不可恢复")
    log(f"     5. 输出命名: 原名 rby1 前插入 filted / ee_filted")

    return results


# ════════════════════════════════════════════════════════════
#  阶段2：合并数据集
# ════════════════════════════════════════════════════════════
def stage2_merge(processed_paths):
    log("\n" + "=" * 60)
    log("【阶段2·合并数据集】")
    log("  对应脚本: scripts/merge_lerobot_fast.py")
    log("=" * 60)

    # 合并水管数据（模拟 v4 + v5 两个子集）
    water_path = processed_paths["water_hose_insertion"]
    tgt_name = f"lerobot_260710_water_hose_insertion_v5_merged_rby1_{NUM_EPISODES*2}s"
    tgt_path = os.path.join(BASE_DIR, "mock_merged", tgt_name)

    log(f"  合并子集1: {os.path.basename(water_path)}")
    log(f"  合并子集2: (模拟第二批同源数据)")
    log(f"  目标路径: {tgt_path}")

    # 读取子集 meta
    with open(os.path.join(water_path, "meta", "info.json"), encoding="utf-8") as f:
        meta1 = json.load(f)

    total_ep = meta1["total_episodes"] * 2  # 模拟两个子集
    total_fr = meta1["total_frames"] * 2

    # 创建合并目录
    for sub in ["data", "videos", "meta"]:
        os.makedirs(os.path.join(tgt_path, sub), exist_ok=True)

    merged_meta = {
        "total_episodes": total_ep,
        "total_frames": total_fr,
        "state_dim": meta1["state_dim"],
        "action_dim": meta1["action_dim"],
        "sub_datasets": ["v4_filted", "v5_filted"],
        "repo_id": "merged_water_hose",
    }
    with open(os.path.join(tgt_path, "meta", "info.json"), "w", encoding="utf-8") as f:
        json.dump(merged_meta, f, indent=2)

    log(f"\n  合并后总条数: {total_ep}")
    log(f"  合并后总帧数: {total_fr}")
    log(f"  ✅ 校验通过: {total_ep} == {meta1['total_episodes']} + {meta1['total_episodes']}")

    log(f"\n  📌 阶段2要点:")
    log(f"     1. --tgt_path 必须事先不存在，由脚本创建")
    log(f"     2. 合并后核对总条数 == 各子集之和")
    log(f"     3. 仅用一批数据时可跳过本步")

    return {
        "water_hose_merged": tgt_path,
        "mainboard_merged": processed_paths["cable_connect_mainboard"],  # 主板不合并
    }


# ════════════════════════════════════════════════════════════
#  阶段3：编写训练 config
# ════════════════════════════════════════════════════════════
def stage3_config(merged_paths):
    log("\n" + "=" * 60)
    log("【阶段3·编写训练 config】")
    log("  对应文件: src/openpi/training/config.py → TrainConfig")
    log("=" * 60)

    # ── 水管任务 (joint + RTC) ──
    water_name = (f"pi05_rby1_water_hole_insertion_v5_mengfan_{NUM_EPISODES*2}s"
                  f"_from_base_bimodal_rtc_delta_absgripper_260710")
    water_cfg = {
        "name": water_name,
        "exp_name": "finetune",
        "task": "water_hose_insertion",
        "control_mode": "joint",
        "rtc_enabled": True,
        "model": {
            "action_horizon": 30,
            "pi05": True,
            "max_delay": 8,
            "delay_sampling": "bimodal",
            "delay_sampling_temperature": 1.0,
            "delay_sampling_second_peak": 6,
            "delay_sampling_second_peak_width": 1.5,
            "delay_sampling_second_peak_weight": 0.5,
            "rtc_loss_scale_mode": "batch",
            "rtc_loss_scale_cap": None,
        },
        "data": {
            "repo_id": merged_paths["water_hose_merged"],
            "arm_joint_mask": "make_bool_mask(7, -7, 1, -1)",
            "default_prompt": "insert the right water hose into the hole",
            "exclude_torso": True,
            "cameras": ["cam_high_left", "cam_high_right",
                        "cam_left_wrist", "cam_right_wrist"],
        },
        "assets": {
            "assets_dir": os.path.dirname(merged_paths["water_hose_merged"]),
            "asset_id": os.path.basename(merged_paths["water_hose_merged"]),
        },
        "weight_loader": {"path": "/tmp/pi05_base/params/"},
        "num_train_steps": 100_000,
        "keep_period": 10_000,
        "lr_schedule": {"warmup_steps": 1000, "peak_lr": 2e-5,
                        "decay_steps": 50_000, "decay_lr": 2e-6},
        "batch_size": 32,
        "log_interval": 50,
        "num_workers": 16,
        "fsdp_devices": 8,
    }

    # ── 主板任务 (EE, 无 RTC) ──
    mb_name = (f"pi05_rby1_cable_connect_mainboard_v4_{NUM_EPISODES}s"
               f"_from_base_ee_delta_euler_260710")
    mb_cfg = {
        "name": mb_name,
        "exp_name": "finetune",
        "task": "cable_connect_mainboard",
        "control_mode": "ee",
        "rtc_enabled": False,
        "model": {
            "action_dim": 32,
            "action_horizon": 30,
            "pi05": True,
        },
        "data": {
            "repo_id": merged_paths["mainboard_merged"],
            "default_prompt": "cable mainboard insertion",
            "action_space": "ee",
            "ee_pose_repr": "euler",
            "arm_mode": "left",
            "use_delta_actions": True,
            "use_cam_high_right": False,
        },
        "assets": {
            "assets_dir": os.path.dirname(merged_paths["mainboard_merged"]),
            "asset_id": os.path.basename(merged_paths["mainboard_merged"]),
        },
        "weight_loader": {"path": "/tmp/pi05_base/params/"},
        "num_train_steps": 100_000,
        "keep_period": 10_000,
        "lr_schedule": {"warmup_steps": 1000, "peak_lr": 2e-5,
                        "decay_steps": 50_000, "decay_lr": 2e-6},
        "batch_size": 32,
        "log_interval": 50,
        "num_workers": 16,
        "fsdp_devices": 8,
    }

    configs = {"water_hose": water_cfg, "mainboard": mb_cfg}
    cfg_path = os.path.join(BASE_DIR, "training_configs.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)

    log(f"\n  ✅ 水管 config:")
    log(f"     name: {water_name}")
    log(f"     joint+RTC | 右手 | action_horizon=30 | steps=100000")
    log(f"     arm_joint_mask: make_bool_mask(7,-7,1,-1) → 左臂左爪屏蔽")
    log(f"     RTC: bimodal, max_delay=8")

    log(f"\n  ✅ 主板 config:")
    log(f"     name: {mb_name}")
    log(f"     EE(无RTC) | 左手 | action_dim=32 | action_space=ee")
    log(f"     ee_pose_repr=euler | use_delta_actions=True")
    log(f"     arm_mode=left → 右臂屏蔽")

    log(f"\n  configs 已保存: {cfg_path}")

    log(f"\n  📌 阶段3要点:")
    log(f"     1. config name 必须与 --config-name 一字不差")
    log(f"     2. 水管=joint+RTC(右手); 主板=EE(左手,无RTC)")
    log(f"     3. assets 指向 norm: assets_dir/asset_id/norm_stats.json")
    log(f"     4. from base: 10万步/1万一存; from ckpt: 2万步/1千一存")
    log(f"     5. decay_steps = num_train_steps / 2")
    log(f"     6. 命名: pi05_机器人_任务_版本_条数s_from_来源_特性_日期")

    return configs


# ════════════════════════════════════════════════════════════
#  阶段4：计算 norm
# ════════════════════════════════════════════════════════════
def stage4_norm(configs):
    log("\n" + "=" * 60)
    log("【阶段4·计算归一化统计量 norm】")
    log("  对应脚本: scripts/compute_norm_states_fast.py")
    log("=" * 60)

    norm_paths = {}

    for task_key, torso_skip in [("water_hose", 6), ("mainboard", 0)]:
        cfg = configs[task_key]
        dataset_path = cfg["data"]["repo_id"]
        meta_file = os.path.join(dataset_path, "meta", "info.json")

        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)

        raw_dim = meta["state_dim"]
        eff_dim = raw_dim - torso_skip
        total_frames = meta["total_frames"]

        log(f"\n  --- {task_key} ---")
        log(f"  config: {cfg['name']}")
        log(f"  dataset: {os.path.basename(dataset_path)}")
        log(f"  --torso: {'6 (joint模式必须)' if torso_skip > 0 else '不加 (EE模式禁止)'}")
        log(f"  原始state维度: {raw_dim}, 有效维度: {eff_dim}")
        log(f"  总帧数: {total_frames}")

        # 模拟统计量
        random.seed(42 + (1 if torso_skip else 0))
        state_mean = [round(random.uniform(-0.5, 0.5), 4) for _ in range(eff_dim)]
        state_std  = [round(random.uniform(0.05, 0.3), 4) for _ in range(eff_dim)]
        action_mean = state_mean[:]
        action_std  = state_std[:]

        norm_stats = {
            "state": {
                "mean": state_mean,
                "std": state_std,
                "min": [round(m - 3*s, 4) for m, s in zip(state_mean, state_std)],
                "max": [round(m + 3*s, 4) for m, s in zip(state_mean, state_std)],
                "median": state_mean,
            },
            "action": {
                "mean": action_mean,
                "std": action_std,
                "min": [round(m - 3*s, 4) for m, s in zip(action_mean, action_std)],
                "max": [round(m + 3*s, 4) for m, s in zip(action_mean, action_std)],
                "median": action_mean,
            },
            "num_frames": total_frames,
            "torso_skipped": torso_skip,
            "config_name": cfg["name"],
        }

        # 写入 assets_dir/asset_id/norm_stats.json
        norm_dir = os.path.join(cfg["assets"]["assets_dir"], cfg["assets"]["asset_id"])
        os.makedirs(norm_dir, exist_ok=True)
        norm_path = os.path.join(norm_dir, "norm_stats.json")
        with open(norm_path, "w", encoding="utf-8") as f:
            json.dump(norm_stats, f, indent=2)

        log(f"  ✅ norm_stats.json 已生成: {norm_path}")
        log(f"     state mean[0:3]: {state_mean[:3]}")
        log(f"     state std[0:3]:  {state_std[:3]}")
        norm_paths[task_key] = norm_path

    log(f"\n  📌 阶段4要点:")
    log(f"     1. ⚠️ joint模式(水管)→必须加 --torso 6, 跳过前6维躯干")
    log(f"     2. ⚠️ EE模式(主板)→禁止加 --torso 6")
    log(f"     3. 夹爪越界(0~0.1外)→加 --correct 修正(直接改数据文件)")
    log(f"     4. norm_stats.json 存于 assets_dir/asset_id/ 下")
    log(f"     5. 训练与推理必须使用同一份 norm")
    log(f"     6. 复用旧 norm 可跳过本步，只需 assets 指向正确")
    log(f"     7. ⚠️ 此处最易出错！加错或漏加都会导致模型训坏！")

    return norm_paths


# ════════════════════════════════════════════════════════════
#  阶段5：启动训练
# ════════════════════════════════════════════════════════════
def cosine_lr(step, warmup, peak_lr, decay_steps, decay_lr, total):
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = min((step - warmup) / max(decay_steps - warmup, 1), 1.0)
    return decay_lr + 0.5 * (peak_lr - decay_lr) * (1 + math.cos(math.pi * progress))


def stage5_train(configs):
    log("\n" + "=" * 60)
    log("【阶段5·启动训练】")
    log("  对应脚本: scripts/train.py")
    log("=" * 60)

    ckpt_dirs = {}

    for task_key, label in [("water_hose", "水管插入(joint+RTC)"),
                             ("mainboard", "主板插线(EE,无RTC)")]:
        cfg = configs[task_key]
        num_steps = cfg["num_train_steps"]
        keep_period = cfg["keep_period"]
        warmup = cfg["lr_schedule"]["warmup_steps"]
        peak_lr = cfg["lr_schedule"]["peak_lr"]
        decay_steps = cfg["lr_schedule"]["decay_steps"]
        decay_lr = cfg["lr_schedule"]["decay_lr"]

        ckpt_base = os.path.join(BASE_DIR, "mock_checkpoints",
                                 cfg["name"], "finetune", "0")
        os.makedirs(ckpt_base, exist_ok=True)

        log(f"\n  --- {label} ---")
        log(f"  config:       {cfg['name']}")
        log(f"  总步数:       {num_steps:,}")
        log(f"  保存周期:     每 {keep_period:,} 步")
        log(f"  batch_size:   {cfg['batch_size']}")
        log(f"  初始权重:     {cfg['weight_loader']['path']}")
        log(f"  控制模式:     {cfg['control_mode']}")
        log(f"  RTC:          {cfg['rtc_enabled']}")
        log(f"  示教数据:     {NUM_EPISODES}条({'合并' if task_key=='water_hose' else '单批'})")
        log(f"  checkpoint:   {ckpt_base}")

        # 采样关键步数（不跑满10万步）
        demo_steps = sorted(set(
            list(range(0, 500, 50)) +
            list(range(0, num_steps + 1, keep_period)) +
            [num_steps]
        ))
        demo_steps = [s for s in demo_steps if s <= num_steps]

        # 模拟 loss 曲线
        random.seed(123)
        loss_log = []
        initial_loss = 2.0
        final_loss = 0.15

        for step in demo_steps:
            lr = cosine_lr(step, warmup, peak_lr, decay_steps, decay_lr, num_steps)
            if step < warmup:
                progress = step / max(warmup, 1)
                base_loss = initial_loss - (initial_loss - 1.0) * progress * 0.5
            else:
                progress = min((step - warmup) / max(num_steps - warmup, 1), 1.0)
                base_loss = 1.0 + (final_loss - 1.0) * (0.5 * (1 - math.cos(math.pi * progress)))
            noise = random.gauss(0, 0.01)
            loss = max(0.01, base_loss + noise)
            loss_log.append((step, round(loss, 6), lr))

        # 打印部分日志
        log(f"\n  {'step':>8s}  {'loss':>10s}  {'lr':>12s}")
        log(f"  {'----':>8s}  {'----':>10s}  {'----':>12s}")
        for step, loss, lr in loss_log[:12]:
            log(f"  {step:>8d}  {loss:>10.6f}  {lr:>12.2e}")
        if len(loss_log) > 17:
            log(f"  {'...':>8s}")
            for step, loss, lr in loss_log[-5:]:
                log(f"  {step:>8d}  {loss:>10.6f}  {lr:>12.2e}")

        # 模拟保存 checkpoint
        ckpt_steps = list(range(keep_period, num_steps + 1, keep_period))
        log(f"\n  📦 保存 {len(ckpt_steps)} 个 checkpoint:")
        for cs in ckpt_steps:
            ckpt_dir = os.path.join(ckpt_base, str(cs))
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_loss = next((l for s, l, _ in reversed(loss_log) if s <= cs), 0.15)
            with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"step": cs, "config": cfg["name"], "loss": ckpt_loss}, f, indent=2)
            if cs <= keep_period * 3 or cs >= num_steps:
                log(f"     step {cs:>7d} → {ckpt_dir}")
        if len(ckpt_steps) > 6:
            log(f"     ... ({len(ckpt_steps) - 6} 个中间 checkpoint)")

        # 写 loss.log
        log_path = os.path.join(ckpt_base, "loss.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"# Training log for {cfg['name']}\n")
            f.write(f"# Total steps: {num_steps}, Batch size: {cfg['batch_size']}\n")
            f.write(f"{'step':>8s}  {'loss':>10s}  {'lr':>12s}\n")
            for step, loss, lr in loss_log:
                f.write(f"{step:>8d}  {loss:>10.6f}  {lr:>12.2e}\n")

        log(f"\n  ✅ 训练完成!")
        log(f"     loss.log: {log_path}")
        log(f"     最终 loss: {loss_log[-1][1]:.6f}")
        log(f"     checkpoint 总数: {len(ckpt_steps)}")

        ckpt_dirs[task_key] = ckpt_base

    log(f"\n  📌 阶段5要点:")
    log(f"     1. from base: 10万步/1万一存; from ckpt: 2万步/1千一存")
    log(f"     2. --overwrite 清空同名目录重训; --resume 断点续训")
    log(f"     3. XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 限制GPU显存")
    log(f"     4. loss 每50步打印，整体应下降，无尖峰毛刺")
    log(f"     5. checkpoint 存于 checkpoint_base_dir/config名/exp-name/步数/")
    log(f"     6. fsdp_devices=8 → 8卡 FSDP 并行训练")
    log(f"     7. 完整命令:")
    log(f"        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \\")
    log(f"        python scripts/train.py config名 --exp-name=0 --overwrite")

    return ckpt_dirs


# ════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log("=" * 60)
    log("  pi05 微调全流程模拟 (纯标准库版)")
    log("  模拟 200~400 条示教数据的完整训练流水线")
    log("  任务: 水管插入(joint+RTC) + 主板插线(EE)")
    log("=" * 60)
    log(f"\n本次模拟示教条数: {NUM_EPISODES}")
    log(f"State维度: {STATE_DIM} (躯干{TORSO_DIM}+关节{JOINT_DIM}+夹爪{GRIPPER_DIM})")

    # 五阶段顺序执行
    processed = stage1_preprocess()
    merged = stage2_merge(processed)
    configs = stage3_config(merged)
    norm_paths = stage4_norm(configs)
    ckpt_dirs = stage5_train(configs)

    # 保存流水线状态
    state = {
        "num_episodes": NUM_EPISODES,
        "water_hose_processed": processed["water_hose_insertion"],
        "mainboard_processed": processed["cable_connect_mainboard"],
        "water_hose_merged": merged["water_hose_merged"],
        "mainboard_merged": merged["mainboard_merged"],
        "water_config_name": configs["water_hose"]["name"],
        "mainboard_config_name": configs["mainboard"]["name"],
        "water_norm_path": norm_paths["water_hose"],
        "mainboard_norm_path": norm_paths["mainboard"],
        "water_ckpt_dir": ckpt_dirs["water_hose"],
        "mainboard_ckpt_dir": ckpt_dirs["mainboard"],
    }
    with open(os.path.join(BASE_DIR, "pipeline_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    log("\n" + "=" * 60)
    log("  🎉 全部 5 个阶段模拟完成！")
    log("=" * 60)
    log("\n  生成的关键产物:")
    log("    mock_processed/         前处理后的数据集")
    log("    mock_merged/            合并后的数据集")
    log("    training_configs.json   两个任务的训练配置")
    log("    norm_stats.json         归一化统计量(各数据集目录下)")
    log("    mock_checkpoints/       训练产出的 checkpoint + loss.log")
    log("    pipeline_state.json     流水线状态")

    # 写日志文件
    log_path = os.path.join(BASE_DIR, "pipeline_output.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES))
    print("\n".join(LOG_LINES))
    print(f"\n📄 完整日志已保存: {log_path}")
