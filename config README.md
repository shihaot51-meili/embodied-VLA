[config_README.md]
## Config 中 `model` 模块详解

### 1. 水管任务的 model 模块（joint + RTC）

```json
"model": {
    "action_horizon": 30,
    "pi05": true,
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

| 参数 | 值 | 含义 |
|------|-----|------|
| `action_horizon` | 30 | 模型一次预测未来 30 步动作（往前看 30 帧的轨迹） |
| `pi05` | true | 使用 pi05 模型（而非 pi0），pi05 是改进版本 |
| `max_delay` | 8 | RTC 最大延迟帧数，模拟推理时最多延迟 8 帧 |
| `delay_sampling` | "bimodal" | 延迟采样方式：双峰分布（两个峰值） |
| `delay_sampling_temperature` | 1.0 | 采样温度，控制分布的"尖锐程度"（1.0=标准） |
| `delay_sampling_second_peak` | 6 | 第二个峰值位置在第 6 帧 |
| `delay_sampling_second_peak_width` | 1.5 | 第二峰的宽度（标准差），越大越平缓 |
| `delay_sampling_second_peak_weight` | 0.5 | 第二峰的权重（0.5=两峰各占一半概率） |
| `rtc_loss_scale_mode` | "batch" | RTC 损失缩放模式：按 batch 维度缩放 |
| `rtc_loss_scale_cap` | null | 损失缩放上限（null=不设上限） |

### 2. 主板任务的 model 模块（EE，无 RTC）

```json
"model": {
    "action_dim": 32,
    "action_horizon": 30,
    "pi05": true
}
```

| 参数 | 值 | 含义 |
|------|-----|------|
| `action_dim` | 32 | 单步动作向量维度（EE 模式必须显式指定） |
| `action_horizon` | 30 | 同上，预测未来 30 步 |
| `pi05` | true | 使用 pi05 模型 |

### 3. 参数分类详解

#### 3.1 基础参数（两个任务都有）

| 参数 | 作用 |
|------|------|
| `action_horizon` | **预测时域**。模型不只预测下一步动作，而是预测未来 30 步的完整轨迹。值越大，模型"看得越远"，但计算量也越大 |
| `pi05` | **模型版本开关**。pi05 相比 pi0 有架构改进，设为 true 启用 |

#### 3.2 EE 专属参数

| 参数 | 作用 |
|------|------|
| `action_dim` | **动作维度**。EE 模式动作空间复杂，必须显式告诉模型输出层大小。Joint 模式可自动推断，所以水管任务不写 |

#### 3.3 RTC 参数（水管任务专属，共 7 个）

**RTC = Real-Time Compensation（实时延迟补偿）**，是一种训练技巧，让模型适应推理时的通信延迟。

**为什么需要 RTC？**
```
训练时: 模型看到第 t 帧观测 → 输出第 t 帧动作（无延迟）
推理时: 模型看到第 t 帧观测 → 但通信有延迟 → 实际执行时已是第 t+δ 帧
         ↑ 训练和推理的延迟不一致，模型性能下降
```

**RTC 的做法**：训练时人为模拟延迟，让模型学会"看到旧观测也能输出合理动作"。

**双峰延迟采样（bimodal）详解**：
```
延迟帧数的概率分布:

概率
 │    第一峰              第二峰
 │     ╱╲                  ╱╲
 │    ╱  ╲                ╱  ╲
 │   ╱    ╲              ╱    ╲
 │  ╱      ╲            ╱      ╲
 │ ╱        ╲          ╱        ╲
 │╱          ╲________╱          ╲
 └──┬─────────┬──────┬──────────────┬──→ 延迟帧数
    0         3      6              8
                        max_delay=8

第一峰: 在低延迟附近（~0-2帧），模拟正常情况
第二峰: 在第6帧附近，模拟较大延迟情况
权重各50%，让模型同时适应低延迟和高延迟
```

| RTC 参数 | 作用 |
|----------|------|
| `max_delay` | 最大允许延迟 8 帧，超过 8 帧不采样 |
| `delay_sampling` | "bimodal" = 双峰分布采样 |
| `delay_sampling_temperature` | 1.0 = 标准温度，控制分布尖锐度 |
| `delay_sampling_second_peak` | 第二峰中心在第 6 帧 |
| `delay_sampling_second_peak_width` | 1.5 = 第二峰宽度（标准差） |
| `delay_sampling_second_peak_weight` | 0.5 = 第二峰占总概率 50% |
| `rtc_loss_scale_mode` | "batch" = 按批次缩放 RTC 损失 |
| `rtc_loss_scale_cap` | null = 不限制损失缩放上限 |

### 4. 两个任务 model 模块对比

| 维度 | 水管（joint+RTC） | 主板（EE） |
|------|-------------------|-----------|
| action_horizon | 30 | 30 |
| pi05 | true | true |
| action_dim | 不写（自动推断） | 32（显式） |
| RTC | ✅ 7 个参数 | ❌ 无 |
| 参数总数 | 10 个 | 3 个 |

### 5. 一句话总结

`model` 模块定义模型架构和行为：`action_horizon` 控制预测多远、`pi05` 选模型版本、`action_dim` 指定输出维度（EE 必写）、RTC 参数组（水管专属）通过双峰延迟采样让模型适应推理时的通信延迟。主板任务不需要 RTC，所以 model 模块极简。

## Config 中 `data` 模块详解

### 1. 水管任务的 data 模块（joint 模式）

```json
"data": {
    "repo_id": "C:\\...\\mock_merged\\lerobot_260710_water_hose_insertion_v5_merged_rby1_742s",
    "arm_joint_mask": "make_bool_mask(7, -7, 1, -1)",
    "prompt_from_task": false,
    "default_prompt": "insert the right water hose into the hole",
    "exclude_torso": true,
    "cameras": ["cam_high_left", "cam_high_right", "cam_left_wrist", "cam_right_wrist"]
}
```

### 2. 主板任务的 data 模块（EE 模式）

```json
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

### 3. 逐参数详解

#### 3.1 两个任务共有的参数

| 参数 | 水管值 | 主板值 | 含义 |
|------|--------|--------|------|
| `repo_id` | 合并后数据集路径 | 前处理后数据集路径 | **训练数据的路径**。水管用阶段2合并后的数据，主板用阶段1前处理后的数据（未合并） |
| `prompt_from_task` | false | false | **是否从任务名自动生成 prompt**。false = 使用 `default_prompt` 的固定文本 |
| `default_prompt` | "insert the right water hose into the hole" | "cable mainboard insertion" | **语言指令**。告诉模型"做什么任务"，是 VLA 模型的语言输入 |

**`prompt_from_task` 详解**：
```
true:  prompt = task 字段值（如 "water_hose_insertion"）
false: prompt = default_prompt 的自定义文本（更自然、更详细）
```

#### 3.2 水管任务专属参数（joint 模式）

| 参数 | 值 | 含义 |
|------|-----|------|
| `arm_joint_mask` | "make_bool_mask(7, -7, 1, -1)" | **关节掩码**：右臂7维启用、左臂7维屏蔽、右爪启用、左爪屏蔽 → 只用右手 |
| `exclude_torso` | true | **排除躯干**：数据保留 22 维但训练时跳过前 6 维躯干关节 |
| `cameras` | 4 个相机 | **使用的相机列表**：左高、右高、左腕、右腕 |

**`cameras` 详解**：
```
cam_high_left   → 左上方俯视相机（全局视野）
cam_high_right  → 右上方俯视相机（全局视野）
cam_left_wrist  → 左手腕相机（第一人称近景）
cam_right_wrist → 右手腕相机（第一人称近景）

水管任务用全部 4 个相机
```

#### 3.3 主板任务专属参数（EE 模式）

| 参数 | 值 | 含义 |
|------|-----|------|
| `action_space` | "ee" | **动作空间**：末端执行器空间（而非关节空间） |
| `ee_pose_repr` | "euler" | **姿态表示**：欧拉角（roll/pitch/yaw） |
| `arm_mode` | "left" | **使用手臂**：只用左臂（对应水管的 arm_joint_mask） |
| `use_delta_actions` | true | **增量动作**：xyz/euler 输出增量，夹爪输出绝对值 |
| `use_cam_high_right` | false | **禁用右高相机**：只用 3 个相机（左高+左腕+右腕） |

**`arm_mode` vs `arm_joint_mask` 对比**：
```
水管（joint）: arm_joint_mask = make_bool_mask(7,-7,1,-1)  → 逐维度掩码
主板（EE）:   arm_mode = "left"                            → 整臂选择

两种方式效果相同（选一只手），但表达方式不同：
- joint 模式在关节维度上做 mask（因为要精确控制每个关节）
- EE 模式直接选左/右臂（因为输出的是末端位姿，不需要逐关节控制）
```

**`use_cam_high_right: false` 详解**：
```
主板任务只用左臂插线，右高相机视角可能被遮挡或无用
→ 禁用右高相机，减少输入冗余，节省计算
```

### 4. 两个任务 data 模块对比

| 维度 | 水管（joint） | 主板（EE） |
|------|--------------|-----------|
| 数据路径 | 合并后（阶段2输出） | 前处理后（阶段1输出） |
| 语言指令 | "insert the right water hose into the hole" | "cable mainboard insertion" |
| 手臂选择 | `arm_joint_mask`（逐维度掩码） | `arm_mode: "left"`（整臂选择） |
| 躯干处理 | `exclude_torso: true` | 无（EE 已无躯干） |
| 动作空间 | 默认（joint） | `action_space: "ee"` |
| 姿态表示 | 无 | `ee_pose_repr: "euler"` |
| 增量动作 | 无 | `use_delta_actions: true` |
| 相机数量 | 4 个（全开） | 3 个（禁用右高） |

### 5. data 模块的作用总结

`data` 模块回答了训练数据的**四个核心问题**：

```
1. 数据在哪？        → repo_id（路径）
2. 做什么任务？      → default_prompt（语言指令）
3. 用哪些自由度？    → arm_joint_mask / arm_mode（手臂选择）+ exclude_torso（躯干）
4. 动作怎么表示？    → action_space / ee_pose_repr / use_delta_actions（EE 专属）
                     + cameras（视觉输入）
```

### 6. 一句话总结

`data` 模块配置训练数据的来源和格式：`repo_id` 指定数据路径，`default_prompt` 提供语言指令，`arm_joint_mask`/`arm_mode` 选择用哪只手，`exclude_torso` 控制躯干去留，EE 模式还额外用 `action_space`/`ee_pose_repr`/`use_delta_actions` 定义动作表示方式，`cameras` 指定视觉输入。两个任务因控制模式不同，data 模块参数差异显著。

## Config中assets（归一化资产）模块详解

### 1. assets 模块长什么样

```json
"assets": {
    "assets_dir": "C:\\...\\mock_merged",
    "asset_id": "lerobot_260710_water_hose_insertion_v5_merged_rby1_742s"
}
```

| 字段 | 作用 | 说明 |
|------|------|------|
| `assets_dir` | **norm 资产所在目录** | norm_stats.json 的父目录 |
| `asset_id` | **资产 ID（子目录名）** | 通常与数据集同名 |

**完整路径拼接**：`assets_dir / asset_id / norm_stats.json`

```
例如水管任务:
  assets_dir = C:\...\mock_merged
  asset_id   = lerobot_260710_water_hose_insertion_v5_merged_rby1_742s
  → norm_stats.json 完整路径:
    C:\...\mock_merged\lerobot_260710_water_hose_insertion_v5_merged_rby1_742s\norm_stats.json
```

### 2. norm_stats.json 是什么

这是阶段4（`simulate_norm.py`）的产物，内容是数据集中 state 和 action 的**统计量**：

```json
{
    "state": {
        "mean":   [0.12, -0.34, 0.56, ...],   // 各维度均值
        "std":    [0.15, 0.22, 0.08, ...],    // 各维度标准差
        "min":    [-0.33, -1.0, 0.32, ...],   // 各维度最小值
        "max":    [0.57, 0.32, 0.80, ...],    // 各维度最大值
        "median": [0.12, -0.35, 0.55, ...]    // 各维度中位数
    },
    "action": {
        "mean":   [...],
        "std":    [...],
        "min":    [...],
        "max":    [...],
        "median": [...]
    },
    "num_frames": 74200,
    "torso_skipped": 6,
    "config_name": "pi05_rby1_water_hole_insertion_v5_mengfan_742s_..."
}
```

### 3. 为什么需要归一化

神经网络对输入数据的**尺度**非常敏感：

```
不归一化的情况:
  关节角度:  [-0.5, 1.2, -0.3, ...]     ← 范围约 ±2
  夹爪开合:  [0.09, 0.05, 0.08, ...]    ← 范围 0~0.1
  躯干关节:  [50, 200, -100, ...]       ← 范围可达几百

  → 数值差异巨大，梯度被大值主导，小值学不动
  → 训练不稳定，收敛慢

归一化后:
  所有维度 → 均值≈0, 标准差≈1
  → 各维度尺度一致，梯度均衡
  → 训练稳定，收敛快
```

**归一化公式**：
```
归一化:   x_norm = (x - mean) / std
反归一化: x = x_norm * std + mean

训练时: 原始 state/action → 减均值除标准差 → 输入模型
推理时: 模型输出 → 乘标准差加均值 → 还原为真实动作
```

### 4. 两个任务的 assets 差异

| 维度 | 水管任务（joint） | 主板任务（EE） |
|------|------------------|---------------|
| `assets_dir` | `mock_merged`（合并后数据父目录） | `mock_processed`（前处理后数据父目录） |
| `asset_id` | 合并后数据集名 | 前处理后数据集名 |
| norm 计算时 `--torso` | **必须加 6**（跳过躯干） | **禁止加**（EE 已无躯干） |
| state 统计维度 | 22 - 6 = 16 维 | 16 维（原始就是 16） |

**为什么水管要跳过躯干 6 维**：
```
水管数据是 22 维: [躯干6 | 右臂7 | 左臂7 | 右爪1 | 左爪1]
                    ↑ 这 6 维不参与训练（exclude_torso=true）
                    
如果统计 norm 时不跳过躯干:
  → 躯干的均值/方差被算进去
  → 归一化时躯干维度被错误缩放
  → 污染后续手臂维度的统计量
  → 模型训坏！

所以 norm 计算时必须 --torso 6，只统计第 7~22 维
```

### 5. assets 路径的默认规则

```python
# simulate_config.py 中的路径生成逻辑
assets_dir = os.path.dirname(dataset_path)   # 数据集的父目录
asset_id   = os.path.basename(dataset_path)  # 数据集名

# → norm_stats.json 默认存在数据集自身目录下
# → assets_dir/asset_id/norm_stats.json = 数据集目录/norm_stats.json
```

**复用旧 norm 的情况**：
- 如果新数据集与旧数据集分布相似，可以复用旧 norm
- 只需把 `assets_dir` 和 `asset_id` 指向旧 norm 所在位置
- 跳过阶段4（不重新计算 norm）

### 6. 核心原则

```
┌──────────────────────────────────────────────────────┐
│  ⚠️ 训练与推理必须使用同一份 norm_stats.json          │
│                                                      │
│  训练时: 用 norm 归一化 → 模型学习归一化空间中的映射    │
│  推理时: 用同一份 norm 反归一化 → 还原为真实动作       │
│                                                      │
│  如果训练和推理用了不同的 norm:                        │
│    → 归一化/反归一化不匹配                             │
│    → 模型输出完全错误                                  │
│    → 机器人乱动！                                     │
└──────────────────────────────────────────────────────┘
```

### 7. 一句话总结

`assets` 模块指向归一化统计量文件 `norm_stats.json` 的位置（`assets_dir/asset_id/norm_stats.json`），该文件包含 state/action 的均值/方差/中位数等统计量，用于训练时归一化输入、推理时反归一化输出。**训练和推理必须用同一份 norm**，水管任务计算 norm 时必须跳过躯干 6 维（`--torso 6`），EE 任务禁止跳过。复用旧 norm 时只需修改 assets 路径，可跳过阶段4。

## config的weight_loader（初始权重来源）模块详解

### 1. weight_loader 模块长什么样

```json
"weight_loader": {
    "type": "CheckpointWeightLoader",
    "path": "/tmp/pi05_base/params/"
}
```

| 字段 | 作用 | 说明 |
|------|------|------|
| `type` | **加载器类型** | "CheckpointWeightLoader" = 从 checkpoint 文件加载预训练权重 |
| `path` | **权重文件路径** | 指向存放模型参数的目录 |

### 2. 两种权重来源

| 模式 | path 值 | 含义 | 训练步数 | 存 checkpoint 周期 |
|------|---------|------|----------|-------------------|
| **from base** | `/tmp/pi05_base/params/` | 从 pi05 基础模型开始训练 | 100,000 步 | 每 10,000 步存一次 |
| **from ckpt** | `/home/appuser/NFS_Share/.../params/` | 从已有 checkpoint 继续训练 | 20,000 步 | 每 1,000 步存一次 |

### 3. 两种模式的区别

#### from base（从基础模型训练）

```
pi05 base 模型（通用预训练）
  ↓ 加载权重
微调训练（100,000 步）
  ↓
任务专用模型

特点:
  - 基础模型在海量通用机器人数据上预训练，具备通用操作能力
  - 微调使其专精特定任务（如水管插入、主板插线）
  - 训练步数多（10万步），因为要从通用能力逐步收敛到特定任务
  - 每 1 万步存一次 checkpoint，共 10 个 checkpoint
```

#### from ckpt（从已有 checkpoint 继续训练）

```
已有 checkpoint（之前训练的某个版本）
  ↓ 加载权重
继续训练（20,000 步）
  ↓
更新后的模型

特点:
  - checkpoint 已经在类似任务上训练过，已有一定能力
  - 只需少量微调（2万步）即可适应新数据或改进
  - 每 1 千步存一次 checkpoint，共 20 个 checkpoint（更密集保存）
  - 适用于：新增数据后继续训练、调整模型表现
```

### 4. 为什么两种模式训练步数差 5 倍

```
from base:
  起点: 通用模型（对特定任务一无所知）
  需要学习: 从零学习任务特定的操作模式
  → 需要大量步数（10万步）才能收敛

from ckpt:
  起点: 已在类似任务上训练过的模型（已掌握基本操作）
  需要学习: 只需微调适应新数据/小改进
  → 少量步数（2万步）即可收敛
```

### 5. checkpoint 存储周期差异的原因

| 模式 | keep_period | 总 checkpoint 数 | 原因 |
|------|------------|-----------------|------|
| from base | 10,000 步 | 10 个 | 训练长，每万步存一次足够观察进展 |
| from ckpt | 1,000 步 | 20 个 | 训练短，密集存储以便挑选最佳 checkpoint |

```
from base 的 checkpoint:
  10K ── 20K ── 30K ── 40K ── 50K ── 60K ── 70K ── 80K ── 90K ── 100K
   1     2     3      4      5      6      7      8      9      10

from ckpt 的 checkpoint:
  1K ── 2K ── 3K ── 4K ── 5K ── ... ── 20K
   1    2    3    4    5          ...   20
```

### 6. 路径对应的实际位置

| 路径 | 位置 | 说明 |
|------|------|------|
| `/tmp/pi05_base/params/` | 容器内临时目录 | pi05 基础模型权重，每次容器启动时加载 |
| `/home/appuser/NFS_Share/.../params/` | NFS 共享存储 | 已训练 checkpoint 的持久化存储，跨容器共享 |

```
容器内路径结构:
/app/openpi-torch/                    ← 训练代码
/tmp/pi05_base/params/                 ← pi05 base 权重（from base）
/home/appuser/NFS_Share/model/checkpoints/  ← checkpoint 产出目录
/home/appuser/NFS_Share/.../params/    ← 已有 checkpoint（from ckpt）
```

### 7. 训练超参如何配合 weight_loader

```json
// from base
{
    "weight_loader": { "path": "/tmp/pi05_base/params/" },
    "num_train_steps": 100000,      ← 多训
    "keep_period": 10000,           ← 少存
    "lr_schedule": {
        "warmup_steps": 1000,
        "peak_lr": 2e-5,
        "decay_steps": 50000,       ← = steps/2
        "decay_lr": 2e-6
    }
}

// from ckpt
{
    "weight_loader": { "path": "/home/appuser/NFS_Share/.../params/" },
    "num_train_steps": 20000,       ← 少训
    "keep_period": 1000,            ← 多存
    "lr_schedule": {
        "warmup_steps": 1000,
        "peak_lr": 2e-5,
        "decay_steps": 10000,       ← = steps/2
        "decay_lr": 2e-6
    }
}
```

**规律**：
- `decay_steps` 始终取 `num_train_steps` 的一半
- `peak_lr` 和 `warmup_steps` 两种模式相同
- `keep_period` 与 `num_train_steps` 成反比（训练短→存更密）

### 8. 一句话总结

`weight_loader` 模块指定训练的初始权重来源：`from base` 从 pi05 基础模型开始（10 万步，每万步存一次），适用于全新任务训练；`from ckpt` 从已有 checkpoint 继续（2 万步，每千步存一次），适用于增量改进。两种模式的学习率峰值和预热步数相同，但训练步数差 5 倍，因为基础模型需要更多步数收敛到特定任务，而 checkpoint 已有基础只需少量微调。

## 模块六：训练超参详解

### 1. 完整配置

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

---

### 2. 逐参数详解

#### 2.1 `num_train_steps`（总训练步数）

| 模式 | 值 | 含义 |
|------|-----|------|
| from base | 100,000 | 从基础模型训练 10 万步 |
| from ckpt | 20,000 | 从 checkpoint 训练 2 万步 |

**什么是"一步"**：
```
一步 = 处理一个 batch 的数据 → 前向传播 → 计算 loss → 反向传播 → 更新权重

batch_size=32 → 每步处理 32 条示教数据
100,000 步 → 共处理 100,000 × 32 = 3,200,000 条数据样本
```

**为什么 from base 是 from ckpt 的 5 倍**：
- from base：模型从通用能力开始，需要大量步数收敛到特定任务
- from ckpt：模型已有任务基础，只需少量微调

---

#### 2.2 `keep_period`（checkpoint 存储周期）

| 模式 | 值 | 含义 |
|------|-----|------|
| from base | 10,000 | 每 1 万步存一次 checkpoint |
| from ckpt | 1,000 | 每 1 千步存一次 |

```
from base 的 checkpoint 时间线:
  10K ── 20K ── 30K ── 40K ── 50K ── 60K ── 70K ── 80K ── 90K ── 100K
   ①     ②     ③     ④     ⑤     ⑥     ⑦     ⑧     ⑨     ⑩
   共 10 个 checkpoint

from ckpt 的 checkpoint 时间线:
  1K ── 2K ── 3K ── 4K ── 5K ── ... ── 20K
   ①    ②    ③    ④    ⑤          ⑳
   共 20 个 checkpoint
```

**为什么 from ckpt 存得更密**：
- from ckpt 只训 2 万步，每千步存一次才能有足够 checkpoint 供挑选
- from base 训 10 万步，每万步存一次已有 10 个 checkpoint，足够观察

---

#### 2.3 `lr_schedule`（学习率调度）

```json
"lr_schedule": {
    "type": "CosineDecaySchedule",   // 余弦衰减调度
    "warmup_steps": 1000,            // 预热步数
    "peak_lr": 2e-5,                 // 峰值学习率
    "decay_steps": 50000,            // 衰减步数
    "decay_lr": 2e-6                 // 最终学习率
}
```

**学习率曲线**：
```
学习率 lr
  │
2e-5 ┤            ╭─────── peak_lr（峰值）
     │           ╱  ╲
     │          ╱    ╲
     │         ╱      ╲         ← CosineDecay（余弦衰减）
     │        ╱        ╲
     │       ╱          ╲
2e-6 ┤______╱            ╲________ decay_lr（最终值）
     │     ╱              ╲
     │    ╱  warmup        ╲
     │   ╱  (线性升温)       ╲
  0  ┤──╱                    ╲────────────
     └──┬────┬─────────┬──────┬────────────┬──→ step
        0   1000      50000              100000
            │           │                   │
         warmup结束   decay结束          训练结束
```

**三个阶段**：

| 阶段 | 步数范围 | 学习率变化 | 作用 |
|------|---------|-----------|------|
| **Warmup（预热）** | 0 → 1000 | 0 → 2e-5（线性升温） | 避免初始学习率过大导致训练崩溃，让模型逐步适应 |
| **Cosine Decay（余弦衰减）** | 1000 → 50000 | 2e-5 → 2e-6（余弦曲线下降） | 逐步降低学习率，让模型在后期精细调整 |
| **稳定阶段** | 50000 → 100000 | 保持 2e-6 | 学习率已很低，模型微调收敛 |

**各参数详解**：

| 参数 | 值 | 含义 |
|------|-----|------|
| `type` | "CosineDecaySchedule" | 余弦衰减策略（比线性衰减更平滑） |
| `warmup_steps` | 1000 | 前 1000 步线性升温，防止初期不稳定 |
| `peak_lr` | 2e-5 | 最大学习率，warmup 结束后达到 |
| `decay_steps` | 50000 | 从 peak_lr 衰减到 decay_lr 用 50000 步（= num_train_steps / 2） |
| `decay_lr` | 2e-6 | 最终学习率，= peak_lr / 10 |

**为什么用余弦衰减而非线性衰减**：
```
线性衰减:  \________  （前期降太快，后期太平）
余弦衰减:  ╲___        （前期降慢，中期加速，后期平滑）
            ↑ 更好的训练曲线，loss 下降更稳定
```

**from base vs from ckpt 的 lr_schedule**：

| 参数 | from base | from ckpt |
|------|-----------|-----------|
| warmup_steps | 1000 | 1000 |
| peak_lr | 2e-5 | 2e-5 |
| decay_steps | 50,000（=100K/2） | 10,000（=20K/2） |
| decay_lr | 2e-6 | 2e-6 |

**规律**：`decay_steps` 始终 = `num_train_steps / 2`，其余参数两种模式相同。

---

#### 2.4 `batch_size`（批大小）

```json
"batch_size": 32
```

| 方面 | 说明 |
|------|------|
| 值 | 32 |
| 含义 | 每步处理 32 条示教数据 |
| 北京集群范围 | 32~64（根据 GPU 显存调整） |
| 影响 | 越大→梯度越稳定但显存占用越高；越小→梯度噪声大但训练快 |

```
batch_size=32 的含义:
  每步从数据集中取 32 条示教轨迹
  → 32 条轨迹的 loss 取平均
  → 用平均梯度更新模型权重
```

---

#### 2.5 `log_interval`（日志间隔）

```json
"log_interval": 50
```

| 方面 | 说明 |
|------|------|
| 值 | 50 |
| 含义 | 每 50 步打印一次 loss 到日志 |
| 输出位置 | `checkpoint_base_dir/config名/exp-name/loss.log` |

```
训练日志示例:
  step 0      loss 2.001736    lr 2.00E-008   ← 第 0 步
  step 50     loss 1.876543    lr 4.00E-008   ← 第 50 步
  step 100    loss 1.723456    lr 8.00E-008   ← 第 100 步
  ...
  step 100000 loss 0.150611    lr 2.00E-006   ← 第 100000 步
```

**判断训练是否正常**：loss 应整体呈下降趋势，无偶发尖峰或毛刺。

---

#### 2.6 `num_workers`（数据加载线程数）

```json
"num_workers": 16
```

| 方面 | 说明 |
|------|------|
| 值 | 16 |
| 含义 | 用 16 个线程并行加载训练数据 |
| 作用 | 避免数据加载成为训练瓶颈（GPU 等数据） |

```
无 num_workers:
  GPU 训练 → 等数据 → 训练 → 等数据 → 训练  ← GPU 空闲浪费

有 num_workers=16:
  16 个线程预取数据 ──→ 数据队列 ──→ GPU 持续训练  ← GPU 不等待
```

---

#### 2.7 `fsdp_devices`（FSDP 并行卡数）

```json
"fsdp_devices": 8
```

| 方面 | 说明 |
|------|------|
| 值 | 8 |
| 含义 | 用 8 张 GPU 做 FSDP 并行训练 |
| FSDP | Fully Sharded Data Parallel（全分片数据并行） |

**FSDP 原理**：
```
普通 DataParallel:
  GPU0: 完整模型 + batch_0
  GPU1: 完整模型 + batch_1    ← 每张卡都存完整模型，显存浪费
  ...
  GPU7: 完整模型 + batch_7

FSDP:
  GPU0: 模型分片0 + batch_0    ← 模型被切片分布到各卡
  GPU1: 模型分片1 + batch_1    ← 显存效率高，可训更大模型
  ...
  GPU7: 模型分片7 + batch_7
```

**为什么用 8 卡**：pi05 模型较大，单卡显存不够，需要 8 卡 FSDP 分摊。

---

### 3. 训练超参全景对比

| 参数 | from base | from ckpt | 说明 |
|------|-----------|-----------|------|
| `num_train_steps` | 100,000 | 20,000 | 总步数，base 是 ckpt 的 5 倍 |
| `keep_period` | 10,000 | 1,000 | 存 checkpoint 周期 |
| `warmup_steps` | 1,000 | 1,000 | 预热步数（相同） |
| `peak_lr` | 2e-5 | 2e-5 | 峰值学习率（相同） |
| `decay_steps` | 50,000 | 10,000 | = num_train_steps / 2 |
| `decay_lr` | 2e-6 | 2e-6 | 最终学习率（相同） |
| `batch_size` | 32 | 32 | 批大小（相同） |
| `log_interval` | 50 | 50 | 日志间隔（相同） |
| `num_workers` | 16 | 16 | 数据加载线程（相同） |
| `fsdp_devices` | 8 | 8 | 并行卡数（相同） |

### 4. 训练过程模拟（from base）

```
step 0      loss 2.001736    lr 2.00E-008   ← warmup 开始，lr 极小
step 1000   loss 1.523456    lr 2.00E-005   ← warmup 结束，lr 达到峰值
step 10000  loss 1.000927    lr 1.85E-005   ← 余弦衰减中，loss 持续下降
step 50000  loss 0.623451    lr 1.00E-005   ← 衰减中点，lr = (peak+decay)/2
step 50000+ loss 0.623451    lr 2.00E-006   ← 衰减结束，lr 降到最低
step 100000 loss 0.150611    lr 2.00E-006   ← 训练结束，loss 收敛

checkpoint 产出:
  10K/  20K/  30K/  40K/  50K/  60K/  70K/  80K/  90K/  100K/
   ①    ②    ③    ④    ⑤    ⑥    ⑦    ⑧    ⑨    ⑩
```

### 5. 一句话总结

训练超参模块控制"怎么训"：`num_train_steps` 决定训练多久（base 10万/ckpt 2万），`keep_period` 决定多久存一次 checkpoint，`lr_schedule` 用余弦衰减策略（warmup 1000步升温到 2e-5 → 50000步衰减到 2e-6），`batch_size=32` 每步处理 32 条数据，`fsdp_devices=8` 用 8 卡并行训练，`num_workers=16` 并行加载数据，`log_interval=50` 每 50 步打印 loss。from base 和 from ckpt 的核心差异是训练步数（5倍）和存储周期（10倍），学习率峰值和预热步数相同。

## 注释:学习率越高越好吗？—— 不是，学习率是一把双刃剑

### 1. 学习率是什么

学习率（Learning Rate, lr）控制每一步更新权重的**幅度**：

```
新权重 = 旧权重 - lr × 梯度
                   ↑
              学习率决定"走多大一步"
```

### 2. 学习率过高 vs 过低 vs 合适

```
学习率过高:
  loss
   │  ╲    ╱╲    ╱╲     ← 来回震荡，无法收敛
   │   ╲  ╱  ╲  ╱  ╲
   │    ╲╱    ╲╱    ╲
   │                   可能发散到 NaN
   └──────────────────────→ step

学习率过低:
  loss
   │
   │
   │
   │___________________    ← 下降极慢，训练效率低
   └──────────────────────→ step
   10万步才降了一点点，浪费算力

学习率合适:
  loss
   │ ╲
   │  ╲
   │   ╲___
   │       ╲___
   │           ╲________    ← 平滑下降，高效收敛
   └──────────────────────→ step
```

### 3. 为什么本项目用 2e-5 而不是更高

| 学习率 | 效果 | 适合场景 |
|--------|------|----------|
| 1e-4 ~ 1e-3 | 太高 | 从头训练小模型 |
| **2e-5** | **合适** | **微调大模型（本项目）** |
| 1e-6 ~ 1e-5 | 太低 | 极精细微调 |

**微调大模型为什么学习率要小**：

```
pi05 是一个在海量数据上预训练的大模型，权重已经很"好"了
→ 学习率太大会破坏已学到的通用能力
→ 只需小幅调整（2e-5）让它适应特定任务

类比:
  从头训练 = 在白纸上画画 → 可以大笔挥洒（大学习率）
  微调     = 在名画上修改 → 必须小心翼翼（小学习率）
```

### 4. 为什么还要 warmup（预热）

```
lr
 │
 │        ╭─── peak_lr = 2e-5
 │       ╱
 │      ╱
 │     ╱
 │    ╱  ← warmup: 线性升温
 │   ╱     从 0 逐步升到 2e-5
 │  ╱      防止初始几步梯度太大导致崩溃
 │ ╱
 │╱
 └──┬──────────────────→ step
    0    1000
         warmup结束
```

**为什么不能直接从 2e-5 开始**：
```
训练刚开始时:
  - 模型还没适应新任务的数据分布
  - 前几步的梯度方向可能很不准确
  - 如果直接用大学习率 → 一步走太远 → 破坏预训练权重

warmup 的作用:
  - 前 1000 步用极小学习率（0 → 2e-5 线性升温）
  - 让模型"试探性"地开始学习
  - 确认梯度方向合理后再加大步伐
```

### 5. 为什么后期要衰减到 2e-6

```
训练前期（lr=2e-5）:
  - 模型离最优解远，需要大步走
  - 快速接近最优解区域

训练后期（lr=2e-6）:
  - 模型已接近最优解
  - 大步走会"跨过"最优解来回震荡
  - 小步走才能精细收敛到最优点

类比:
  前期 = 从北京到上海 → 坐飞机（大学习率）
  后期 = 找到具体门牌号 → 步行（小学习率）
```

### 6. 学习率与 loss 的关系（本项目实际曲线）

```
step 0      loss 2.001736    lr 2.00E-008   ← warmup，lr 极小
step 1000   loss 1.523456    lr 2.00E-005   ← warmup 结束，lr 达峰
step 10000  loss 1.000927    lr 1.85E-005   ← lr 开始衰减，loss 快速下降
step 50000  loss 0.623451    lr 1.00E-005   ← lr 衰减中点，loss 稳步下降
step 100000 loss 0.150611    lr 2.00E-006   ← lr 最低，loss 收敛

观察:
  - lr 高时（2e-5）→ loss 下降快（每万步降 ~0.1）
  - lr 低时（2e-6）→ loss 下降慢（每万步降 ~0.02）
  - 这正是余弦衰减的设计意图：前期快学，后期精调
```

### 7. 如果学习率设错了会怎样

| 问题 | 症状 | 原因 |
|------|------|------|
| 学习率太高 | loss 震荡或不下降，甚至 NaN | 每步走太远，跨过最优解 |
| 学习率太低 | loss 几乎不下降 | 每步走太小，训练效率极低 |
| 无 warmup | 前几步 loss 突然飙升 | 初始梯度方向不准 + 大学习率 |
| 无衰减 | 后期 loss 震荡不收敛 | 接近最优解时步子仍太大 |

### 8. 一句话总结

**学习率不是越高越好**。学习率是"每步走多大"的控制旋钮：太高会震荡发散（跨过最优解），太低会收敛极慢（浪费算力）。本项目用 `peak_lr=2e-5`（微调大模型的合理值），配合 warmup（前 1000 步从 0 升到峰值，防止初期崩溃）和余弦衰减（从 2e-5 逐步降到 2e-6，前期快学后期精调），实现 loss 从 ~2.0 平滑下降到 ~0.15 的理想训练曲线。
