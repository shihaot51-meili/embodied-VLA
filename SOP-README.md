
## SOP 文件解析 — 容器名与环境设置

感谢你提供了完整的 SOP 文件内容！根据这份 `SOP_docker_pi05jax_training.txt`，以下是关键信息：

### 1. 容器名

从 SOP 中的命令提示符可以看出容器名：

```
(openpi) root'0d944f555c2e:app#' 
```

- **容器 ID**：`0d944f555c2e`（Docker 自动生成的短 ID）
- **用户**：`root`
- **工作目录**：`/app`（即 `/app/openpi-torch` 的上级）

**SOP 中没有给出容器的人类可读名称（如 `--name xxx`）**，使用的是 Docker 自动生成的 ID。实际使用时通过 `docker ps` 查看容器 ID 或名称即可。

### 2. 两套环境配置（SOP 中有差异）

SOP 中提供了**两套环境变量配置**，对应不同的集群：

#### 配置 A — 通用配置（可能是上海/其他集群）

```bash
cd /app/openpi-torch
source /app/openpi-torch/.venv/bin/activate
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OPENPI_DATA_HOME=/group-volume/openpi-cache/
export JAX_PLATFORMS=cuda
export XLA_FLAGS=--xla_gpu_cuda_data_dir=./jax_cache
export TMPDIR=./jax_cache/
unset LEROBOT_HOME
export HF_LEROBOT_HOME=/group-volume/huggingface-cache/lerobot
```

#### 配置 B — 北京集群配置

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

**两套配置的差异**：

| 环境变量 | 配置 A（通用） | 配置 B（北京集群） |
|---------|---------------|-------------------|
| `OPENPI_DATA_HOME` | `/group-volume/openpi-cache/` | `/home/appuser/Robot_TF/model/pretrain_model/tokenizers` |
| `OPENPI_LOCAL_ASSETS_DIR` | 未设置 | `/home/appuser/Robot_TF/model/pretrain_model` |
| `LEROBOT_HOME` | `unset`（清除） | 未设置 |
| `HF_LEROBOT_HOME` | `/group-volume/huggingface-cache/lerobot` | 未设置 |

### 3. 4 卡训练的特殊配置（优化项）

SOP 中 4 卡训练有一套更复杂的 XLA 优化参数：

```bash
export XLA_FLAGS="\
--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_enable_pipelined_all_gather=true \
--xla_gpu_enable_pipelined_reduce_scatter=true \
--xla_gpu_enable_pipelined_all_reduce=true \
--xla_gpu_enable_pipelined_collectives=false \
--xla_gpu_all_gather_combine_threshold_bytes=134217728 \
--xla_gpu_reduce_scatter_combine_threshold_bytes=134217728 \
--xla_gpu_all_reduce_combine_threshold_bytes=33554432"
```

这些是 **XLA GPU 通信优化参数**，用于加速多卡 FSDP 训练：

| 参数 | 作用 |
|------|------|
| `latency_hiding_scheduler` | 隐藏通信延迟 |
| `pipelined_all_gather` | 流水线化 All-Gather 通信 |
| `pipelined_reduce_scatter` | 流水线化 Reduce-Scatter 通信 |
| `pipelined_all_reduce` | 流水线化 All-Reduce 通信 |
| `combine_threshold_bytes` | 通信操作合并阈值（128MB/32MB） |

### 4. PGLE 缓存（首次训练编译优化）

首次训练时启用 PGLE（Persistent GPU Compilation Cache），大幅减少 JAX 编译时间：

```bash
# 首次训练（启用 PGLE 编译缓存）
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR=/app/jax_pgle_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_LOG_COMPILES=1
export JAX_ENABLE_PGLE=true
export JAX_PGLE_PROFILING_RUNS=3

# 第二次起训练（复用已录好的 cache，跳过 PGLE）
unset JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS
unset JAX_ENABLE_PGLE
unset JAX_PGLE_PROFILING_RUNS
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR=/app/jax_pgle_cache
export JAX_COMPILATION_CACHE_EXPECT_PGLE=yes
```

**流程**：
```
第一次训练:
  PGLE profiling 3 次 → 生成编译缓存 → 正式训练
  ↑ 编译时间长（可能几十分钟），但只做一次

第二次及以后:
  直接复用缓存 → 跳过 PGLE → 快速启动训练
```

### 5. config 中的关键路径差异（SOP vs 模拟项目）

SOP 中的真实 config 比模拟项目多了一些字段：

| 字段 | SOP 真实值 | 模拟项目值 | 说明 |
|------|-----------|-----------|------|
| `checkpoint_base_dir` | `/home/appuser/NFS_Share/model/.../checkpoints/` 或 `/mnt/ckpt_ram/checkpoints/` | 无（模拟） | 真实 checkpoint 保存路径 |
| `assets_base_dir` | `/home/appuser/Robot_TF/dataset/.../assets` | 无（模拟） | 无用字段（SOP 注释说"没用"） |
| `wandb_enabled` | `False` | 无 | 关闭 WandB 日志 |
| `repack_transforms` | 相机映射配置 | 无 | 数据打包时的字段重映射 |

### 6. 4 卡训练的 config 差异

```python
# 4 卡训练时
fsdp_devices=4,          # 从 8 改为 4
save_interval=5000,      # 注意：SOP 中用 save_interval 而非 keep_period
checkpoint_base_dir="/mnt/ckpt_ram/checkpoints/",  # 用 RAM 盘加速
```

### 7. SOP 中的其他实用命令

| 功能 | 命令 |
|------|------|
| **开环评估** | `serve_policy.py` + `openloop_eval_client.py` |
| **敏感性分析** | `pi05_patch_sensitivity.py`（16×16 网格遮挡检测） |
| **提取初始位姿** | `extract_initial_states.py` |
| **切割 DAgger 数据** | `cut_lerobot_data.py --cut-seconds 2.0` |
| **可视化图像变换** | `test_final_transforms_visualization.py` |
| **修复 git proxy** | unset http_proxy/https_proxy + git config --unset |

### 8. 完整环境设置流程（基于 SOP 整理）

```bash
# ── 1. 进入容器 ──
docker exec -it 0d944f555c2e bash  # 或用 docker ps 查到的容器名

# ── 2. 初始化环境（北京集群）──
cd /app/openpi-torch
source /app/openpi-torch/.venv/bin/activate
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OPENPI_DATA_HOME=/home/appuser/Robot_TF/model/pretrain_model/tokenizers
export OPENPI_LOCAL_ASSETS_DIR="/home/appuser/Robot_TF/model/pretrain_model"
export JAX_PLATFORMS=cuda
export TMPDIR=./jax_cache/

# ── 3. 首次训练加 PGLE 优化（第二次起跳过）──
mkdir -p jax_pgle_cache
export XLA_FLAGS="--xla_gpu_cuda_data_dir=./jax_cache \
  --xla_gpu_enable_latency_hiding_scheduler=true \
  --xla_gpu_enable_pipelined_all_gather=true \
  --xla_gpu_enable_pipelined_reduce_scatter=true \
  --xla_gpu_enable_pipelined_all_reduce=true \
  --xla_gpu_all_gather_combine_threshold_bytes=134217728 \
  --xla_gpu_reduce_scatter_combine_threshold_bytes=134217728 \
  --xla_gpu_all_reduce_combine_threshold_bytes=33554432"
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR=/app/jax_pgle_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_LOG_COMPILES=1
export JAX_ENABLE_PGLE=true
export JAX_PGLE_PROFILING_RUNS=3

# ── 4. 启动训练 ──
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py \
  pi05_rby1_water_hole_insertion_v5_mengfan_200s_from_base_bimodal_rtc_delta_absgripper_260712 \
  --exp-name=1 \
  --overwrite
```

### 9. 一句话总结

根据 SOP 文件，**容器名是 Docker 自动生成的 ID（如 `0d944f555c2e`）**，用 `docker ps` 查看。环境设置分两套（通用集群 vs 北京集群），北京集群用 `/home/appuser/Robot_TF/model/pretrain_model/tokenizers` 作为 tokenizer 路径。4 卡训练额外需配置 XLA 通信优化参数和 PGLE 编译缓存（首次编译、后续复用）。SOP 还包含开环评估、敏感性分析、DAgger 数据切割等实用命令。
