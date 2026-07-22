#!/usr/bin/env python3
"""
阶段3：编写训练 config（模拟）
============================================================
对应文档第6节：src/openpi/training/config.py 中的 TrainConfig

在 config.py 中仿照已有条目新增一段 TrainConfig。
本脚本生成两个模拟 config（水管 joint+RTC、主板 EE），
写入 training_configs.json 供后续 norm 和训练脚本读取。

命名规范：pi05_机器人_任务_数据版本_数据条数s_from_初始权重来源_特性标记_日期
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def make_water_hose_config(dataset_path, num_episodes, norm_assets_dir, norm_asset_id,
                           from_base=True):
    """
    水管任务（joint 模式 + RTC）模板。
    - joint 模式：模型直接输出关节角度
    - 只用右手，左臂左爪 mask 为 0
    - 开启 RTC（延迟适应）
    """
    ckpt_path = "/tmp/pi05_base/params/" if from_base else "/home/appuser/NFS_Share/.../params/"
    name = (f"pi05_rby1_water_hole_insertion_v5_mengfan_{num_episodes}s"
            f"_from_{'base' if from_base else 'ckpt'}_bimodal_rtc_delta_absgripper_260710")

    config = {
        # ── 基本信息 ──
        "name": name,
        "exp_name": "finetune",
        "task": "water_hose_insertion",
        "control_mode": "joint",
        "rtc_enabled": True,

        # ── 模型配置 (Pi0Config) ──
        "model": {
            "action_horizon": 30,         # 动作预测时域：预测未来30步动作
            "pi05": True,                 # 使用 pi05 而非 pi0
            # ── RTC 参数（照抄模板，一般不改）──
            "max_delay": 8,               # 最大延迟帧数
            "delay_sampling": "bimodal",  # 双峰延迟采样
            "delay_sampling_temperature": 1.0,
            "delay_sampling_second_peak": 6,
            "delay_sampling_second_peak_width": 1.5,
            "delay_sampling_second_peak_weight": 0.5,
            "rtc_loss_scale_mode": "batch",
            "rtc_loss_scale_cap": None,
        },

        # ── 数据配置 (LeRobotRby1DataConfig) ──
        "data": {
            "repo_id": dataset_path,     # 合并后的数据集路径
            "arm_joint_mask": "make_bool_mask(7, -7, 1, -1)",  # 只动右手
            "prompt_from_task": False,
            "default_prompt": "insert the right water hose into the hole",
            "exclude_torso": True,        # 排除6维躯干关节
            # 相机映射（照抄）
            "cameras": ["cam_high_left", "cam_high_right",
                        "cam_left_wrist", "cam_right_wrist"],
        },

        # ── norm 资产路径 ──
        "assets": {
            "assets_dir": norm_assets_dir,   # assets_dir/asset_id/norm_stats.json
            "asset_id": norm_asset_id,
        },

        # ── 权重加载 ──
        "weight_loader": {
            "type": "CheckpointWeightLoader",
            "path": ckpt_path,               # from base → /tmp/pi05_base/params/
        },

        # ── 训练超参 ──
        "num_train_steps": 100_000 if from_base else 20_000,  # base 10万 / ckpt 2万
        "keep_period": 10_000 if from_base else 1_000,        # base 1万一存 / ckpt 1千一存
        "lr_schedule": {
            "type": "CosineDecaySchedule",
            "warmup_steps": 1000,
            "peak_lr": 2e-5,
            "decay_steps": 50_000,          # 一般取 num_train_steps 的一半
            "decay_lr": 2e-6,
        },
        "batch_size": 32,                   # 北京集群 32~64
        "log_interval": 50,
        "num_workers": 16,
        "fsdp_devices": 8,                   # 8卡 FSDP 并行
    }
    return name, config


def make_mainboard_config(dataset_path, num_episodes, norm_assets_dir, norm_asset_id,
                          from_base=True):
    """
    主板任务（EE 模式，无 RTC）模板。
    - EE 模式：模型输出末端执行器位姿，由机器人解算关节
    - 只用左手，右臂 mask
    - 不开 RTC
    """
    ckpt_path = "/tmp/pi05_base/params/" if from_base else "/home/appuser/NFS_Share/.../params/"
    name = (f"pi05_rby1_cable_connect_mainboard_v4_{num_episodes}s"
            f"_from_{'base' if from_base else 'ckpt'}_ee_delta_euler_260710")

    config = {
        "name": name,
        "exp_name": "finetune",
        "task": "cable_connect_mainboard",
        "control_mode": "ee",
        "rtc_enabled": False,

        "model": {
            "action_dim": 32,               # EE 任务须显式写 32
            "action_horizon": 30,
            "pi05": True,
            # 无 RTC 参数
        },

        "data": {
            "repo_id": dataset_path,
            "prompt_from_task": False,
            "default_prompt": "cable mainboard insertion",
            "action_space": "ee",            # EE 控制模式
            "ee_pose_repr": "euler",         # 姿态用欧拉角
            "arm_mode": "left",              # 只用左臂
            "use_delta_actions": True,       # xyz/euler 增量，夹爪绝对值
            "use_cam_high_right": False,
            "use_quantile_norm": False,
        },

        "assets": {
            "assets_dir": norm_assets_dir,
            "asset_id": norm_asset_id,
        },

        "weight_loader": {
            "type": "CheckpointWeightLoader",
            "path": ckpt_path,
        },

        "num_train_steps": 100_000 if from_base else 20_000,
        "keep_period": 10_000 if from_base else 1_000,
        "lr_schedule": {
            "type": "CosineDecaySchedule",
            "warmup_steps": 1000,
            "peak_lr": 2e-5,
            "decay_steps": 50_000,
            "decay_lr": 2e-6,
        },
        "batch_size": 32,
        "log_interval": 50,
        "num_workers": 16,
        "fsdp_devices": 8,
    }
    return name, config


if __name__ == "__main__":
    state_file = os.path.join(BASE_DIR, "pipeline_state.json")
    with open(state_file) as f:
        state = json.load(f)

    num_ep = state["num_episodes"]

    # ── 水管任务 config ──
    water_dataset = state["water_hose_merged"]
    water_assets_dir = os.path.dirname(water_dataset)   # norm 存数据集父目录下
    water_asset_id = os.path.basename(water_dataset)

    water_name, water_cfg = make_water_hose_config(
        dataset_path=water_dataset,
        num_episodes=num_ep,
        norm_assets_dir=water_assets_dir,
        norm_asset_id=water_asset_id,
        from_base=True,
    )

    # ── 主板任务 config ──
    mainboard_dataset = state["mainboard_merged"]
    mb_assets_dir = os.path.dirname(mainboard_dataset)
    mb_asset_id = os.path.basename(mainboard_dataset)

    mb_name, mb_cfg = make_mainboard_config(
        dataset_path=mainboard_dataset,
        num_episodes=num_ep,
        norm_assets_dir=mb_assets_dir,
        norm_asset_id=mb_asset_id,
        from_base=True,
    )

    # 保存 configs
    configs = {
        "water_hose": {"name": water_name, "config": water_cfg},
        "mainboard":  {"name": mb_name, "config": mb_cfg},
    }
    cfg_path = os.path.join(BASE_DIR, "training_configs.json")
    with open(cfg_path, "w") as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"【阶段3·编写训练 config】")
    print(f"{'='*60}")
    print(f"\n  ✅ 水管任务 config:")
    print(f"     name: {water_name}")
    print(f"     模式: joint + RTC")
    print(f"     action_horizon: 30, batch_size: 32, steps: 100000")
    print(f"     arm_joint_mask: 右手(7,-7,1,-1) 左臂左爪屏蔽")
    print(f"     RTC: bimodal, max_delay=8")

    print(f"\n  ✅ 主板任务 config:")
    print(f"     name: {mb_name}")
    print(f"     模式: EE (无 RTC)")
    print(f"     action_dim: 32, action_space: ee, ee_pose_repr: euler")
    print(f"     arm_mode: left, use_delta_actions: True")
    print(f"     steps: 100000, keep_period: 10000")

    print(f"\n  configs 已保存到: {cfg_path}")

    # 更新 state
    state["water_config_name"] = water_name
    state["mainboard_config_name"] = mb_name
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

    print(f"\n📌 要点回顾：")
    print(f"   1. config name 必须与后续命令 --config-name 一字不差")
    print(f"   2. 水管=joint+RTC(右手), 主板=EE(左手,无RTC)")
    print(f"   3. assets 指向 norm 所在目录: assets_dir/asset_id/norm_stats.json")
    print(f"   4. from base: 10万步/1万一存; from ckpt: 2万步/1千一存")
    print(f"   5. decay_steps 一般取 num_train_steps 的一半")
