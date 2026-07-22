[Uploading README.md…]()
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
