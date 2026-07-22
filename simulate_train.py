#!/usr/bin/env python3
"""
阶段5：启动训练（模拟）
============================================================
对应文档第8节：scripts/train.py

模拟完整训练过程：
  - from base: 10万步，每1万步存一次 checkpoint（共10个）
  - from ckpt: 2万步，每1千步存一次 checkpoint（共20个）
  - 每50步打印一次 loss（模拟余弦衰减 + 随机噪声）
  - loss 整体应呈下降趋势，无偶发尖峰

输出：checkpoint 目录结构 + loss.log
"""

import json
import os
import math
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def cosine_decay_lr(step, warmup, peak_lr, decay_steps, decay_lr, total_steps):
    """模拟余弦衰减学习率。"""
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = min((step - warmup) / (decay_steps - warmup), 1.0)
    return decay_lr + 0.5 * (peak_lr - decay_lr) * (1 + math.cos(math.pi * progress))


def simulate_training(config_name, config, task_label, num_episodes):
    """模拟训练循环。"""
    num_steps = config["num_train_steps"]
    keep_period = config["keep_period"]
    log_interval = config["log_interval"]
    batch_size = config["batch_size"]
    peak_lr = config["lr_schedule"]["peak_lr"]
    warmup = config["lr_schedule"]["warmup_steps"]
    decay_steps = config["lr_schedule"]["decay_steps"]
    decay_lr = config["lr_schedule"]["decay_lr"]

    # checkpoint 目录
    ckpt_base = os.path.join(BASE_DIR, "mock_checkpoints", config_name, "finetune", "0")
    os.makedirs(ckpt_base, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"【阶段5·启动训练】 {task_label}")
    print(f"{'='*60}")
    print(f"  config:       {config_name}")
    print(f"  总步数:       {num_steps:,}")
    print(f"  保存周期:     每 {keep_period:,} 步")
    print(f"  batch_size:   {batch_size}")
    print(f"  初始权重:     {config['weight_loader']['path']}")
    print(f"  RTC:          {config.get('rtc_enabled', False)}")
    print(f"  控制模式:     {config['control_mode']}")
    print(f"  示教数据:     {num_episodes} 条")
    print(f"  checkpoint目录: {ckpt_base}")
    print()

    # ── 模拟训练循环 ──
    # 为了演示速度，我们不跑满全部步数，而是采样关键节点
    # 实际训练中每50步打印一次 loss
    demo_steps = sorted(set(
        list(range(0, 500, 50)) +                     # 前500步密集
        list(range(0, num_steps + 1, keep_period)) +  # 每个 checkpoint 节点
        [num_steps]                                    # 最后一步
    ))
    demo_steps = [s for s in demo_steps if s <= num_steps]

    loss_log = []
    rng = np.random.default_rng(123)

    # 模拟 loss 曲线：从 ~2.0 余弦衰减到 ~0.15 + 噪声
    initial_loss = 2.0
    final_loss = 0.15

    for step in demo_steps:
        lr = cosine_decay_lr(step, warmup, peak_lr, decay_steps, decay_lr, num_steps)

        # 模拟 loss（余弦衰减 + 小噪声）
        if step < warmup:
            progress = step / max(warmup, 1)
            base_loss = initial_loss - (initial_loss - 1.0) * progress * 0.5
        else:
            progress = min((step - warmup) / (num_steps - warmup), 1.0)
            base_loss = 1.0 + (final_loss - 1.0) * (0.5 * (1 - math.cos(math.pi * progress)))

        noise = rng.normal(0, 0.01)
        # 偶尔小波动（但不出现尖峰毛刺）
        loss = max(0.01, base_loss + noise)

        loss_log.append({"step": step, "loss": round(loss, 6), "lr": f"{lr:.2e}"})

    # 打印部分日志
    print(f"  {'step':>8s}  {'loss':>10s}  {'lr':>12s}")
    print(f"  {'----':>8s}  {'----':>10s}  {'----':>12s}")
    for entry in loss_log[:12]:  # 打印前12条
        print(f"  {entry['step']:>8d}  {entry['loss']:>10.6f}  {entry['lr']:>12s}")
    if len(loss_log) > 12:
        print(f"  {'...':>8s}")
        for entry in loss_log[-5:]:
            print(f"  {entry['step']:>8d}  {entry['loss']:>10.6f}  {entry['lr']:>12s}")

    # ── 模拟保存 checkpoint ──
    ckpt_steps = list(range(keep_period, num_steps + 1, keep_period))
    print(f"\n  📦 保存 {len(ckpt_steps)} 个 checkpoint:")
    for cs in ckpt_steps:
        ckpt_dir = os.path.join(ckpt_base, str(cs))
        os.makedirs(ckpt_dir, exist_ok=True)
        # 模拟写入 checkpoint 文件
        with open(os.path.join(ckpt_dir, "checkpoint_meta.json"), "w") as f:
            json.dump({
                "step": cs,
                "config": config_name,
                "loss": next((e["loss"] for e in reversed(loss_log) if e["step"] <= cs), 0.15),
            }, f, indent=2)
        if cs <= keep_period * 3 or cs >= num_steps - keep_period:
            print(f"     step {cs:>7d}/  → {ckpt_dir}")

    if len(ckpt_steps) > 6:
        print(f"     ... ({len(ckpt_steps) - 6} 个中间 checkpoint 省略)")

    # ── 保存 loss.log ──
    log_path = os.path.join(ckpt_base, "loss.log")
    with open(log_path, "w") as f:
        f.write(f"# Training log for {config_name}\n")
        f.write(f"# Total steps: {num_steps}, Batch size: {batch_size}\n")
        f.write(f"{'step':>8s}  {'loss':>10s}  {'lr':>12s}\n")
        for entry in loss_log:
            f.write(f"{entry['step']:>8d}  {entry['loss']:>10.6f}  {entry['lr']:>12s}\n")

    print(f"\n  ✅ 训练完成！")
    print(f"     loss.log: {log_path}")
    print(f"     最终 loss: {loss_log[-1]['loss']:.6f}")
    print(f"     checkpoint 总数: {len(ckpt_steps)}")

    return ckpt_base


if __name__ == "__main__":
    state_file = os.path.join(BASE_DIR, "pipeline_state.json")
    with open(state_file) as f:
        state = json.load(f)

    with open(os.path.join(BASE_DIR, "training_configs.json")) as f:
        configs = json.load(f)

    num_ep = state["num_episodes"]

    # ── 水管任务训练 ──
    water_ckpt = simulate_training(
        config_name=configs["water_hose"]["name"],
        config=configs["water_hose"]["config"],
        task_label="水管插入 (joint + RTC)",
        num_episodes=num_ep,
    )

    # ── 主板任务训练 ──
    mb_ckpt = simulate_training(
        config_name=configs["mainboard"]["name"],
        config=configs["mainboard"]["config"],
        task_label="主板插线 (EE, 无 RTC)",
        num_episodes=num_ep,
    )

    state["water_ckpt_dir"] = water_ckpt
    state["mainboard_ckpt_dir"] = mb_ckpt
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

    print(f"\n{'='*60}")
    print(f"📌 训练阶段要点回顾：")
    print(f"{'='*60}")
    print(f"   1. from base: 10万步, 每1万步存ckpt; from ckpt: 2万步, 每1千步存")
    print(f"   2. --overwrite 清空同名目录重训; --resume 断点续训")
    print(f"   3. XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 限制GPU显存占用")
    print(f"   4. loss 每50步打印，整体应下降，无尖峰毛刺")
    print(f"   5. checkpoint 存于 checkpoint_base_dir/config名/exp-name/步数/")
    print(f"   6. fsdp_devices=8 → 8卡 FSDP 并行训练")
