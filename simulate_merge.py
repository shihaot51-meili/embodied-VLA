#!/usr/bin/env python3
"""
阶段2：合并数据集（模拟）
============================================================
对应文档第5节：scripts/merge_lerobot_fast.py

当训练需要多批数据时，将多个处理后的子数据集合并为一个大集。
本脚本模拟合并过程：读取多个子集的 meta 信息，累加帧数，
生成合并后的目录结构。

注意：--tgt_path 必须事先不存在，由脚本创建。
"""

import json
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def merge(src_paths, tgt_path, repo_id="merged_dataset"):
    print(f"\n{'='*60}")
    print(f"【阶段2·合并数据集】")
    print(f"{'='*60}")

    total_episodes = 0
    total_frames = 0
    merged_meta = {"sub_datasets": [], "repo_id": repo_id}

    for i, src in enumerate(src_paths):
        meta_file = os.path.join(src, "meta", "info.json")
        if not os.path.exists(meta_file):
            print(f"  ⚠️  跳过（找不到 meta）: {src}")
            continue

        with open(meta_file) as f:
            meta = json.load(f)

        ep = meta["total_episodes"]
        fr = meta["total_frames"]
        total_episodes += ep
        total_frames += fr

        merged_meta["sub_datasets"].append({
            "path": os.path.basename(src),
            "episodes": ep,
            "frames": fr,
        })
        print(f"  子集{i+1}: {os.path.basename(src)}  → {ep} 条 / {fr} 帧")

    # 确保目标路径不存在
    if os.path.exists(tgt_path):
        print(f"  ⚠️  目标目录已存在，删除重建: {tgt_path}")
        shutil.rmtree(tgt_path)

    for sub in ["data", "videos", "meta"]:
        os.makedirs(os.path.join(tgt_path, sub), exist_ok=True)

    merged_meta["total_episodes"] = total_episodes
    merged_meta["total_frames"] = total_frames
    merged_meta["state_dim"] = 22
    merged_meta["action_dim"] = 22

    with open(os.path.join(tgt_path, "meta", "info.json"), "w") as f:
        json.dump(merged_meta, f, indent=2)

    print(f"\n  合并后总条数: {total_episodes}")
    print(f"  合并后总帧数: {total_frames}")
    print(f"  输出路径:     {tgt_path}")

    # 核对
    expected = sum(s["episodes"] for s in merged_meta["sub_datasets"])
    assert total_episodes == expected, "合并条数不等于各子集之和！"
    print(f"  ✅ 校验通过：{total_episodes} == {expected}（各子集之和）")

    print(f"\n📌 要点回顾：")
    print(f"   1. tgt_path 必须事先不存在，由脚本创建")
    print(f"   2. 合并后核对总条数 == 各子集之和")
    print(f"   3. 仅用一批数据时可跳过本步")
    return tgt_path


if __name__ == "__main__":
    state_file = os.path.join(BASE_DIR, "pipeline_state.json")
    with open(state_file) as f:
        state = json.load(f)

    # 模拟合并两个子集（这里复用同一路径模拟多个子集）
    src1 = state["water_hose_processed"]
    src2 = state["mainboard_processed"]

    # 合并水管数据（模拟 v4 + v5）
    tgt_water = os.path.join(BASE_DIR, "mock_merged",
                             "lerobot_260710_water_hose_insertion_v5_merged_rby1_400s")
    merge([src1, src1], tgt_water)  # 用同一路径模拟两个子集

    # 更新状态
    state["water_hose_merged"] = tgt_water
    state["mainboard_merged"] = src2  # 主板只一批，不合并
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

    print(f"\n✅ 合并完成，状态已更新到 pipeline_state.json")
