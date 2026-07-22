# pi05 微调全流程模拟 — 说明文档

本仓库模拟了用 200~400 条机器人示教数据微调 pi05 基础模型的完整五阶段流程。每次运行随机生成 200~400 条示教数据，依次走通：**数据前处理 → 数据合并 → 编写 config → 计算 norm → 启动训练**。

## 快速运行

```powershell
# Windows PowerShell（无需 Python，纯 PowerShell 脚本）
powershell -ExecutionPolicy Bypass -File pi05_pipeline_sim.ps1
```

运行后会在当前目录下生成 `mock_processed/`、`mock_merged/`、`mock_checkpoints/` 等目录，以及 `training_configs.json`、`norm_stats.json`、`pipeline_state.json`、`pipeline_output.log` 等文件。

---

## 五阶段详解

### 阶段1：数据前处理

**对应脚本**：`scripts/preprocess_lerobot_data_fast.py`

**做什么**：对原始 lerobot 示教数据进行清洗，核心三步：

| 步骤 | 参数 | 作用 |
|------|------|------|
| 异常值滤除 | `--fix_outliers` | 将超出合理范围的 state/action 值 clip 到 [-3, 3]，原始数据中约 2% 的帧含异常值（50~500），必须清除 |
| 动作对齐 | `--action_to_next_state` | 将 action 赋值为下一帧的 state，使动作序列平滑一致。原始数据中 action 与 state 不对齐（action 含大噪声） |
| 控制空间转换 | `--ee-mode` | 主板任务用：将 state/action 从关节空间转为末端执行器(EE)空间。水管任务不加此参数，保持关节空间 |

**任务固定对应**：
- **水管插入**（water_hose_insertion）：joint 模式，不加 `--ee-mode`
- **主板插线**（cable_connect_mainboard）：EE 模式，加 `--ee-mode`

**注意事项**：
- `--output_dir` 必填！不填会原地修改原始数据且不可恢复
- 输出命名：在原名 rby1 前插入 `filted`（joint）或 `ee_filted`（EE）
- 名称已含 `filted` 的表示处理过，无需重复
- `--set_left_gripper_to 0.09` / `--set_right_gripper_to 0.09` 仅在修复违规采集数据时使用
- 输出目录出现 `data/`、`videos/`、`meta/` 子目录即成功

**模拟输出示例**（本次运行 339 条）：
```
water_hose_insertion: 33900 frames, state_dim=22 (joint 模式)
cable_connect_mainboard: 33900 frames, state_dim=16 (EE 模式，去掉6维躯干)
```

---

### 阶段2：合并数据集

**对应脚本**：`scripts/merge_lerobot_fast.py`

**做什么**：当训练需要多批数据时（如 v4 + v5），将其合并为一个大集。

**关键规则**：
- `--tgt_path` 指向的目录**必须事先不存在**，由脚本创建
- 合并后核对总条数 == 各子集之和
- 仅使用一批数据时可跳过本步

**模拟输出**：
```
Source 1: lerobot_260710_water_hose_insertion_v5_filted_rby1_339s
Source 2: (simulated 2nd batch)
Merged episodes: 678
Merged frames: 67800
[OK] Verified: 678 == 339 + 339
```

---

### 阶段3：编写训练 config

**对应文件**：`src/openpi/training/config.py` 中的 `TrainConfig`

**做什么**：在 config.py 中仿照已有条目新增一段 TrainConfig。

**命名规范**：`pi05_机器人_任务_数据版本_数据条数s_from_初始权重来源_特性标记_日期`

**两个任务的固定差异**：

| 维度 | 水管插入（joint+RTC） | 主板插线（EE） |
|------|----------------------|---------------|
| 控制模式 | joint（直接输出关节角度） | ee（输出末端位姿，机器人解算关节） |
| 使用手臂 | 右手（左臂左爪 mask=0） | 左手（右臂 mask） |
| RTC | 开启（延迟适应技巧） | 不开启 |
| action_dim | 默认 | 显式写 32 |
| arm_joint_mask | `make_bool_mask(7, -7, 1, -1)` | 无（用 arm_mode="left"） |
| action_space | 默认(joint) | "ee" |
| ee_pose_repr | 无 | "euler" |
| use_delta_actions | 无 | True（xyz/euler增量，夹爪绝对值） |
| exclude_torso | True（排除6维躯干） | 无（EE已去掉躯干） |

**RTC 参数**（水管任务照抄即可）：
```python
max_delay=8,
delay_sampling="bimodal",
delay_sampling_temperature=1.0,
delay_sampling_second_peak=6,
delay_sampling_second_peak_width=1.5,
delay_sampling_second_peak_weight=0.5,
rtc_loss_scale_mode="batch",
rtc_loss_scale_cap=None,
```

**训练超参**：
| 参数 | from base | from ckpt |
|------|-----------|-----------|
| num_train_steps | 100,000 | 20,000 |
| keep_period | 10,000 | 1,000 |
| decay_steps | 50,000（=num_train_steps/2） | 10,000 |
| peak_lr | 2e-5 | 2e-5 |
| warmup_steps | 1,000 | 1,000 |
| batch_size | 32（北京集群32~64） | 32 |
| fsdp_devices | 8（8卡并行） | 8 |

**norm 资产路径**：
- `assets_dir/asset_id/norm_stats.json` 为 norm 的完整路径
- 默认存数据集自身目录；复用旧 norm 则填指定位置

**生成的 config 名称示例**：
```
水管: pi05_rby1_water_hole_insertion_v5_mengfan_678s_from_base_bimodal_rtc_delta_absgripper_260710
主板: pi05_rby1_cable_connect_mainboard_v4_339s_from_base_ee_delta_euler_260710
```

---

### 阶段4：计算归一化统计量 norm

**对应脚本**：`scripts/compute_norm_states_fast.py`

**做什么**：根据 config 名定位数据集，统计 state 和 action 的均值/方差/中位数，写出 `norm_stats.json`。

**⚠️ 最易出错的分支（务必注意！）**：

| 任务 | 控制模式 | --torso 参数 | 原因 |
|------|---------|-------------|------|
| 水管插入 | joint | **必须加** `--torso 6` | 跳过前6维躯干关节，否则躯干统计量污染 |
| 主板插线 | EE | **禁止加** `--torso 6` | EE 模式数据已无躯干维度 |

**夹爪越界修正**：
- 正常范围 0~0.1，异常时可达几百至几万
- 追加 `--correct` 由脚本修正（直接修改数据集文件）
- 前处理阶段保留的原始数据即为备份

**核心原则**：
- norm_stats.json 存于 `assets_dir/asset_id/` 下
- **训练与推理必须使用同一份 norm**
- 复用旧 norm 可跳过本步，只需 assets 指向正确位置

**模拟输出**：
```
water_hose: --torso 6, raw_dim=22, effective=16, frames=67800
mainboard:  --torso none, raw_dim=16, effective=16, frames=33900
```

---

### 阶段5：启动训练

**对应脚本**：`scripts/train.py`

**启动命令**：
```bash
# 全新训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py <config名> --exp-name=0 --overwrite

# 断点续训
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py <config名> --exp-name=0 --resume
```

**参数说明**：
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`：限制 JAX 使用 90% GPU 显存
- `--overwrite`：清空同名实验目录后重新开始（首次运行无问题；若有需要的 checkpoint 不要用）
- `--resume`：从已有 checkpoint 继续训练
- `--exp-name`：实验名，区分不同训练运行

**训练过程监控**：
- 每 50 步打印一次 loss（`log_interval=50`）
- 完整 loss.log 保存在 `checkpoint_base_dir/config名/exp-name/` 目录下
- **正常训练 loss 应整体呈下降趋势，无偶发尖峰或毛刺**
- checkpoint 按设定周期出现在 `.../config名/exp-name/步数/` 目录下

**训练时长**：
- from base：约 10 万步，每 1 万步存一次（共 10 个 checkpoint）
- from ckpt：最多约 2 万步，每 1 千步存一次（共 20 个 checkpoint）

**模拟 loss 曲线**（本次运行）：
```
水管任务:
  step 0      loss 2.001736    lr 2.00E-008  (warmup 阶段)
  step 10000  loss 1.000927    lr 1.85E-005  (peak lr 附近)
  step 50000  loss 0.623451    lr 1.00E-005  (衰减中点)
  step 100000 loss 0.150611    lr 2.00E-006  (最终)

主板任务:
  step 0      loss 2.000143    lr 2.00E-008
  step 100000 loss 0.149788    lr 2.00E-006
```

loss 从 ~2.0 平滑下降到 ~0.15，符合正常训练曲线特征。

---

## 环境准备（真实训练时需要）

训练代码在容器内 `/app/openpi-torch` 路径下。每开新终端需执行：

```bash
cd /app/openpi-torch
source /app/openpi-torch/.venv/bin/activate
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OPENPI_DATA_HOME=/home/appuser/Robot_TF/model/pretrain_model/tokenizers
export OPENPI_LOCAL_ASSETS_DIR="/home/appuser/Robot_TF/model/pretrain_model"
export JAX_PLATFORMS=cuda
export XLA_FLAGS=--xla_gpu_cuda_data_dir=./jax_cache
export TMPDIR=./jax_cache/
```

**固定路径**：
- pi05 base 模型权重：`/tmp/pi05_base/params/`
- 数据集：`/home/appuser/Robot_TF/dataset/lerobot/` 或 `/home/appuser/NFS_Share/dataset/lerobot/`
- checkpoint 产出：`/home/appuser/NFS_Share/model/checkpoints/`

---

## 生成的文件结构

```
mock_processed/                          # 阶段1输出
  lerobot_260710_water_hose_insertion_v5_filted_rby1_339s/
    data/episode_000000.json             # 示教数据
    videos/                              # 视频文件
    meta/info.json                       # 元数据
  lerobot_260710_cable_connect_mainboard_v5_ee_filted_rby1_339s/
    ...

mock_merged/                             # 阶段2输出
  lerobot_260710_water_hose_insertion_v5_merged_rby1_678s/
    meta/info.json
    ...

training_configs.json                    # 阶段3输出（两个任务的完整配置）

mock_merged/.../norm_stats.json          # 阶段4输出（norm 统计量）
mock_processed/.../norm_stats.json

mock_checkpoints/                        # 阶段5输出
  pi05_rby1_water_hole_insertion_v5_mengfan_678s_.../
    finetune/0/
      loss.log                           # 训练日志
      10000/checkpoint_meta.json         # checkpoint
      20000/checkpoint_meta.json
      ...
      100000/checkpoint_meta.json

pipeline_state.json                      # 流水线状态（各阶段路径）
pipeline_output.log                      # 完整运行日志
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `pi05_pipeline_sim.ps1` | **主运行脚本**（PowerShell，无依赖），运行此文件执行全流程 |
| `pi05_pipeline_sim.py` | Python 版主脚本（需 Python + 无第三方依赖） |
| `simulate_preprocess.py` | 阶段1单独脚本（Python，需 numpy） |
| `simulate_merge.py` | 阶段2单独脚本 |
| `simulate_config.py` | 阶段3单独脚本 |
| `simulate_norm.py` | 阶段4单独脚本 |
| `simulate_train.py` | 阶段5单独脚本 |
| `run_all.py` | Python 版主控脚本（依次调用上述5个脚本） |

##注释:(1)Joint 模式 vs EE 模式对比，以及与正逆运动学的关系

### 1. 两种控制模式的本质区别

| 维度 | Joint 模式（关节空间） | EE 模式（末端执行器空间） |
|------|----------------------|------------------------|
| **模型输出** | 直接输出各关节角度（如 7 维向量） | 输出末端位姿（xyz 位置 + euler/四元数姿态 + 夹爪） |
| **执行流程** | 模型输出 → 直接发给电机 | 模型输出位姿 → **机器人控制器做逆运动学** → 解算出关节角度 → 发给电机 |
| **动作空间** | 关节角度（弧度） | 末端位姿（位置+姿态） |
| **维度** | 7（单臂关节） | 7（xyz=3 + euler=3 + gripper=1）或更多 |
| **直观性** | 低（关节角度不直观） | 高（位姿直接描述"手往哪放"） |
| **训练难度** | 模型需学习关节协调 | 模型学习笛卡尔空间运动，更自然 |

### 2. 与正逆运动学的关系

**是的，直接相关。** 核心区别就在于**逆运动学（IK）由谁来做**：

```
┌───────────── Joint 模式 ─────────────┐
│                                      │
│  模型 ──→ 关节角度 ──→ 电机          │
│         (直接控制)                    │
│                                      │
│  正运动学(FK)只在观测时用：          │
│  关节角度 → 末端位姿（用于反馈/可视化）│
│                                      │
│  逆运动学(IK)：完全不需要！           │
└──────────────────────────────────────┘

┌───────────── EE 模式 ───────────────┐
│                                      │
│  模型 ──→ 末端位姿 ──→ IK求解 ──→ 关节角度 ──→ 电机
│                      (机器人控制器做)                  │
│                                      │
│  逆运动学(IK)：由机器人控制器实时求解  │
│  正运动学(FK)：模型隐式学习位姿→运动   │
└──────────────────────────────────────┘
```

#### 正运动学（Forward Kinematics, FK）
- **关节角度 → 末端位姿**
- 这是唯一解，计算简单（矩阵乘法）
- 两种模式都可能用到（如计算当前末端位姿作为观测）

#### 逆运动学（Inverse Kinematics, IK）
- **末端位姿 → 关节角度**
- 可能**多解、无解**（冗余自由度问题）
- 计算复杂，需要数值迭代或解析求解
- **EE 模式的核心依赖**：模型只管输出"手放到哪里"，由机器人控制器实时做 IK 解算出关节角度

### 3. 为什么两个任务选不同模式

| | 水管插入（Joint + RTC） | 主板插线（EE） |
|---|---|---|
| **任务特点** | 需要精细的关节级协调，插入过程涉及多关节联动 | 主要关心末端位姿（插线位置和角度），关节如何弯曲不重要 |
| **选择理由** | 直接控制关节避免 IK 求解的不确定性，RTC 延迟适应在关节空间更稳定 | EE 空间更直观，位姿增量（delta）描述插线动作更自然 |
| **delta action** | 关节增量 | `use_delta_actions: True`（xyz/euler 增量 + 夹爪绝对值） |

### 4. 一句话总结

- **Joint 模式**：模型直接输出关节角度，**不需要逆运动学**，控制最直接但学习难度大
- **EE 模式**：模型输出末端位姿，**依赖机器人控制器做逆运动学**解算关节角度，更直观但引入 IK 求解的延迟和不确定性

两者的选择本质上是**把逆运动学的计算放在哪一侧**的问题：Joint 模式让模型自己学会关节协调（绕过 IK），EE 模式把 IK 交给机器人控制器（模型只管位姿）。

## (2)Config 是什么？

在这个 pi05 微调流水线中，**config（训练配置）** 是阶段3的产物，对应真实代码中 `src/openpi/training/config.py` 里的 `TrainConfig`。它是一段 JSON/Python 配置，定义了**如何训练模型**的全部参数，包括模型设置、数据路径、权重加载、训练超参等。

---

### Config 的核心结构

以 `training_configs.json` 为例，每个 config 包含以下几大块：

| 模块 | 作用 | 关键字段 |
|------|------|----------|
| **基本信息** | 标识任务与控制模式 | `name`（命名规范：`pi05_机器人_任务_版本_条数s_from_权重来源_特性_日期`）、`task`、`control_mode` |
| **model** | 模型配置 | `action_horizon`（预测时域=30）、`pi05: true`、RTC 参数（水管任务专属）或 `action_dim: 32`（EE 任务专属） |
| **data** | 数据集配置 | `repo_id`（合并后数据集路径）、`arm_joint_mask`/`arm_mode`（用哪只手）、相机列表、prompt 等 |
| **assets** | norm 归一化资产路径 | `assets_dir/asset_id/norm_stats.json` |
| **weight_loader** | 初始权重来源 | `from base` → `/tmp/pi05_base/params/`；`from ckpt` → NFS 路径 |
| **训练超参** | 训练步数、学习率等 | `num_train_steps`、`keep_period`、`lr_schedule`（CosineDecay）、`batch_size`、`fsdp_devices` |

---

### 两个任务的 Config 差异

| 维度 | 水管插入（joint + RTC） | 主板插线（EE） |
|------|------------------------|---------------|
| 控制模式 | `joint`（直接输出关节角度） | `ee`（输出末端位姿，机器人解算关节） |
| 使用手臂 | 右手（左臂左爪 mask=0） | 左手（`arm_mode="left"`） |
| RTC | ✅ 开启（延迟适应） | ❌ 不开启 |
| action_dim | 默认 | 显式写 32 |
| ee_pose_repr | 无 | `"euler"` |
| use_delta_actions | 无 | `True`（xyz/euler 增量，夹爪绝对值） |
| exclude_torso | `True` | 无（EE 已去掉躯干） |

---

### 训练超参（from base vs from ckpt）

| 参数 | from base | from ckpt |
|------|-----------|-----------|
| num_train_steps | 100,000 | 20,000 |
| keep_period | 10,000 | 1,000 |
| decay_steps | 50,000（=steps/2） | 10,000 |
| peak_lr | 2e-5 | 2e-5 |
| batch_size | 32 | 32 |
| fsdp_devices | 8 | 8 |

---

### 生成方式

`simulate_config.py` 脚本读取 `pipeline_state.json`（前序阶段的输出路径），调用 `make_water_hose_config()` 和 `make_mainboard_config()` 两个模板函数生成配置，写入 `training_configs.json`，供后续的 norm 计算和训练脚本使用。

**关键要点**：config 的 `name` 必须与后续训练命令 `python scripts/train.py <config名>` 中的参数一字不差，是串联整个流水线的核心标识。

## 总Config 各模块详细总结

Config（训练配置）是 pi05 微调流水线阶段3的产物，定义了"怎么训练"的全部参数。以 `training_configs.json` 为例，每个 config 包含以下模块：

### 模块一：基本信息

```json
{
    "name": "pi05_rby1_water_hole_insertion_v5_mengfan_742s_from_base_bimodal_rtc_delta_absgripper_260710",
    "exp_name": "finetune",
    "task": "water_hose_insertion",
    "control_mode": "joint",
    "rtc_enabled": true
}
```

| 字段 | 作用 | 说明 |
|------|------|------|
| `name` | **配置名称**（唯一标识） | 命名规范：`pi05_机器人_任务_版本_条数s_from_权重来源_特性_日期`。必须与训练命令 `--config-name` 一字不差 |
| `exp_name` | **实验名称** | "finetune" = 微调实验，用于区分不同训练运行 |
| `task` | **任务类型** | "water_hose_insertion" / "cable_connect_mainboard" |
| `control_mode` | **控制模式** | "joint"（关节空间）/ "ee"（末端执行器空间） |
| `rtc_enabled` | **是否开启 RTC** | 水管=true，主板=false |

**作用**：标识这个 config 训练什么任务、用什么控制方式，是串联整个流水线的核心 ID。

---

### 模块二：model（模型配置）

```json
"model": {
    "action_horizon": 30,
    "pi05": true,
    "action_dim": 32,          // 仅 EE 任务
    // ── RTC 参数（仅水管任务）──
    "max_delay": 8,
    "delay_sampling": "bimodal",
    "delay_sampling_temperature": 1.0,
    "delay_sampling_second_peak": 6,
    "delay_sampling_second_peak_width": 1.5,
    "delay_sampling_second_peak_weight": 0.5,
    "rtc_loss_scale_mode": "batch",
    "rtc_loss_scale_cap": null
}
```

| 字段 | 作用 | 说明 |
|------|------|------|
| `action_horizon` | **预测时域** | 一次预测未来 30 步动作轨迹，值越大看得越远 |
| `pi05` | **模型版本** | true=用 pi05（pi0 的改进版） |
| `action_dim` | **单步动作维度** | EE 模式必须显式写 32；joint 模式自动推断 |
| `max_delay` | **RTC 最大延迟** | 训练时模拟最多 8 帧延迟 |
| `delay_sampling` | **延迟采样分布** | "bimodal"=双峰分布（低延迟峰 + 高延迟峰） |
| `delay_sampling_temperature` | **采样温度** | 1.0=标准，控制分布尖锐度 |
| `delay_sampling_second_peak` | **第二峰位置** | 第 6 帧，模拟较大延迟 |
| `delay_sampling_second_peak_width` | **第二峰宽度** | 1.5=标准差，越大越平缓 |
| `delay_sampling_second_peak_weight` | **第二峰权重** | 0.5=两峰各占 50% 概率 |
| `rtc_loss_scale_mode` | **RTC 损失缩放** | "batch"=按批次缩放 |
| `rtc_loss_scale_cap` | **损失缩放上限** | null=不设上限 |

**作用**：定义模型架构和行为。`action_horizon` 控制预测多远，`pi05` 选模型版本，`action_dim` 指定输出大小，RTC 参数组通过双峰延迟采样让模型适应推理时的通信延迟。

---

### 模块三：data（数据集配置）

```json
// 水管任务（joint）
"data": {
    "repo_id": "C:\\...\\mock_merged\\lerobot_260710_water_hose_insertion_v5_merged_rby1_742s",
    "arm_joint_mask": "make_bool_mask(7, -7, 1, -1)",
    "prompt_from_task": false,
    "default_prompt": "insert the right water hose into the hole",
    "exclude_torso": true,
    "cameras": ["cam_high_left", "cam_high_right", "cam_left_wrist", "cam_right_wrist"]
}

// 主板任务（EE）
"data": {
    "repo_id": "C:\\...\\mock_processed\\lerobot_260710_cable_connect_mainboard_v5_ee_filted_rby1_371s",
    "prompt_from_task": false,
    "default_prompt": "cable mainboard insertion",
    "action_space": "ee",
    "ee_pose_repr": "euler",
    "arm_mode": "left",
    "use_delta_actions": true,
    "use_cam_high_right": false
}
```

| 字段 | 作用 | 水管值 | 主板值 |
|------|------|--------|--------|
| `repo_id` | **训练数据路径** | 合并后数据集（阶段2输出） | 前处理后数据集（阶段1输出） |
| `prompt_from_task` | **prompt 来源** | false（用 default_prompt） | false |
| `default_prompt` | **语言指令** | "insert the right water hose into the hole" | "cable mainboard insertion" |
| `arm_joint_mask` | **关节掩码**（joint 专属） | make_bool_mask(7,-7,1,-1) → 右手 | 无 |
| `arm_mode` | **手臂选择**（EE 专属） | 无 | "left" → 左手 |
| `exclude_torso` | **排除躯干**（joint 专属） | true（跳过 6 维躯干） | 无（EE 已无躯干） |
| `action_space` | **动作空间**（EE 专属） | 无（默认 joint） | "ee" |
| `ee_pose_repr` | **姿态表示**（EE 专属） | 无 | "euler"（roll/pitch/yaw） |
| `use_delta_actions` | **增量动作**（EE 专属） | 无 | true（xyz/euler 增量，夹爪绝对值） |
| `cameras` | **相机列表** | 4 个全开 | 无（用 use_cam_high_right 控制） |
| `use_cam_high_right` | **禁用右高相机** | 无 | false（只用 3 个相机） |

**作用**：回答训练数据的四个核心问题——数据在哪（repo_id）、做什么任务（default_prompt）、用哪些自由度（arm_joint_mask/arm_mode/exclude_torso）、动作怎么表示（action_space/ee_pose_repr/use_delta_actions）+ 视觉输入（cameras）。

---

### 模块四：assets（归一化资产）

```json
"assets": {
    "assets_dir": "C:\\...\\mock_merged",
    "asset_id": "lerobot_260710_water_hose_insertion_v5_merged_rby1_742s"
}
```

| 字段 | 作用 | 说明 |
|------|------|------|
| `assets_dir` | **norm 资产目录** | norm_stats.json 所在的父目录 |
| `asset_id` | **资产 ID** | 子目录名，与数据集同名 |

**完整路径**：`assets_dir/asset_id/norm_stats.json`

**作用**：指向阶段4计算的归一化统计量文件。训练和推理必须使用同一份 norm，确保 state/action 的均值/方差归一化一致。复用旧 norm 时只需修改此路径。

---

### 模块五：weight_loader（权重加载）

```json
"weight_loader": {
    "type": "CheckpointWeightLoader",
    "path": "/tmp/pi05_base/params/"
}
```

| 字段 | 作用 | 说明 |
|------|------|------|
| `type` | **加载器类型** | "CheckpointWeightLoader" = 从 checkpoint 加载 |
| `path` | **权重路径** | from base → `/tmp/pi05_base/params/`；from ckpt → NFS 路径 |

**作用**：指定初始权重来源。两种模式：
- **from base**：从 pi05 基础模型开始训练（10 万步）
- **from ckpt**：从已有 checkpoint 继续训练（2 万步）

---

### 模块六：训练超参

```json
{
    "num_train_steps": 100000,
    "keep_period": 10000,
    "lr_schedule": {
        "type": "CosineDecaySchedule",
        "warmup_steps": 1000,
        "peak_lr": 2e-5,
        "decay_steps": 50000,
        "decay_lr": 2e-6
    },
    "batch_size": 32,
    "log_interval": 50,
    "num_workers": 16,
    "fsdp_devices": 8
}
```

| 字段 | 作用 | from base | from ckpt |
|------|------|-----------|-----------|
| `num_train_steps` | **总训练步数** | 100,000 | 20,000 |
| `keep_period` | **存 checkpoint 周期** | 每 10,000 步存一次 | 每 1,000 步存一次 |
| `lr_schedule.type` | **学习率调度** | CosineDecaySchedule（余弦衰减） | 同左 |
| `lr_schedule.warmup_steps` | **预热步数** | 1,000（线性升温） | 1,000 |
| `lr_schedule.peak_lr` | **峰值学习率** | 2e-5 | 2e-5 |
| `lr_schedule.decay_steps` | **衰减步数** | 50,000（=steps/2） | 10,000 |
| `lr_schedule.decay_lr` | **最终学习率** | 2e-6 | 2e-6 |
| `batch_size` | **批大小** | 32（北京集群 32~64） | 32 |
| `log_interval` | **日志间隔** | 每 50 步打印 loss | 50 |
| `num_workers` | **数据加载线程** | 16 | 16 |
| `fsdp_devices` | **FSDP 并行卡数** | 8（8 卡 FSDP） | 8 |

**学习率曲线**：
```
lr
2e-5 ┤        ╭─── peak
     │       ╱
     │      ╱
     │     ╱     ← CosineDecay
     │    ╱
2e-6 ┤___╱────────────
     └──┬────────┬────┬──→ step
        1000    50000  100000
       warmup  decay  end
```

**作用**：控制训练过程的所有超参数——训练多久、多久存一次、学习率怎么变化、批大小多大、用几张卡。

---

### 全局总结

```
┌─────────────────────────────────────────────────────────┐
│                      Config 全景图                       │
├──────────────┬──────────────────────────────────────────┤
│  基本信息     │ 我是谁？训练什么任务？用什么控制模式？      │
├──────────────┼──────────────────────────────────────────┤
│  model       │ 模型怎么建？预测多远？输出多大？要不要RTC？  │
├──────────────┼──────────────────────────────────────────┤
│  data        │ 数据在哪？做什么任务？用哪只手？动作怎么表示？│
├──────────────┼──────────────────────────────────────────┤
│  assets      │ 归一化统计量在哪？训练和推理用同一份 norm    │
├──────────────┼──────────────────────────────────────────┤
│  weight_loader│ 初始权重从哪来？base 模型还是旧 checkpoint？│
├──────────────┼──────────────────────────────────────────┤
│  训练超参     │ 训练多久？学习率怎么变？批多大？几张卡？     │
└──────────────┴──────────────────────────────────────────┘
```

**一句话总结**：Config 是一份完整的训练配方，6 个模块分别回答了"训什么任务、模型怎么建、数据怎么读、norm 在哪、权重从哪来、怎么训"六个问题，是串联整个五阶段流水线的核心配置文件。
