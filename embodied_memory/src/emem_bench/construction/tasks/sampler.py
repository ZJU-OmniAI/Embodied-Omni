"""L0/L1 任务采样器

sample_tasks() 是对外的主入口：给定场景 metadata 和层级，
返回去重后的任务列表。
"""

from __future__ import annotations
from typing import Optional

import numpy as np

from ..scene_utils import SceneQuery
from .schema import Task
from .templates import (
    COMPAT_TABLE,
    make_l0_pick_and_place,
    make_l1_clean_and_place,
    make_l1_heat_and_place,
    make_l1_cool_and_place,
    make_l1_pick_two_and_place,
)

# 容器内物体数达到此阈值时视为拥挤，不作为目标容器
_CROWDED_THRESHOLD = 1


def sample_tasks(
    metadata: dict,
    level: str = "L1",
    n_tasks: int = 3,
    seed: Optional[int] = None,
) -> list[Task]:
    """
    从场景元数据采样指定层级的任务。

    采样流程：
      1. 枚举所有候选任务（每个物体最多一个）
      2. 随机打乱后去重（key_object 唯一），取前 n_tasks 个

    Args:
        metadata: AI2-THOR controller.last_event.metadata 或离线 JSON
        level:    "L0"（简单搬运）或 "L1"（带状态变换的复杂任务）
        n_tasks:  目标任务数量（候选不足时返回全部）
        seed:     随机种子，保证可复现
    """
    rng = np.random.RandomState(seed)
    sq = SceneQuery(metadata)
    recep_by_type = sq.receptacles_by_type()

    # 统计每个容器当前已有的物体数量
    recep_load: dict[str, int] = {}
    for o in metadata["objects"]:
        for pid in o.get("parentReceptacles") or []:
            recep_load[pid] = recep_load.get(pid, 0) + 1

    candidates: list[Task] = []

    if level == "L0":
        candidates = _sample_l0(sq, recep_by_type, recep_load, rng)
    elif level == "L1":
        candidates = _sample_l1(sq, recep_by_type, recep_load, rng)

    if not candidates:
        return []

    # 打乱候选池，使结果不依赖枚举顺序
    rng.shuffle(candidates)

    # 去重：同一个物体（key_object）只出现在一个任务里
    selected: list[Task] = []
    used: set[str] = set()
    for c in candidates:
        if c.key_object in used:
            continue
        selected.append(c)
        used.add(c.key_object)
        if len(selected) >= n_tasks:
            break

    return selected


# ─── 内部采样逻辑 ─────────────────────────────────────────────


def _sample_l0(sq, recep_by_type, recep_load, rng) -> list[Task]:
    candidates = []
    for obj in sq.pickupables():
        compat = COMPAT_TABLE.get(obj["objectType"], [])
        for dt in compat:
            for dest in recep_by_type.get(dt, []):
                parent = sq.parent_of(obj)
                if parent and dest["objectId"] != parent["objectId"]:
                    if recep_load.get(dest["objectId"], 0) >= _CROWDED_THRESHOLD:
                        continue
                    task = make_l0_pick_and_place(sq, obj, dest)
                    if task:
                        candidates.append(task)
                        break  # 每个物体只生成一个 L0
    return candidates


def _sample_l1(sq, recep_by_type, recep_load, rng) -> list[Task]:
    candidates = []

    for obj in sq.pickupables():
        compat = COMPAT_TABLE.get(obj["objectType"], [])

        # 收集可行的目标容器（排除拥挤容器）
        possible_dests = []
        for dt in compat:
            for d in recep_by_type.get(dt, []):
                parent = sq.parent_of(obj)
                if parent and d["objectId"] != parent["objectId"]:
                    if recep_load.get(d["objectId"], 0) >= _CROWDED_THRESHOLD:
                        continue
                    possible_dests.append(d)
        if not possible_dests:
            continue

        # 随机选一个目标容器
        dest = possible_dests[rng.randint(len(possible_dests))]

        # 按优先级依次尝试 L1 模板，第一个成功的加入候选池
        # 注意：clean 优先级最高，导致 clean 任务占比偏多
        for factory in [
            make_l1_clean_and_place,
            make_l1_heat_and_place,
            make_l1_cool_and_place,
        ]:
            task = factory(sq, obj, dest)
            if task:
                candidates.append(task)
                break

    # pick_two_and_place：随机抽两个物体，各自独立选目标容器
    # 最多尝试 min(15, len(pickups)) 次
    pickups = sq.pickupables()
    all_receps = [
        r
        for r in sq.receptacles()
        if recep_load.get(r["objectId"], 0) < _CROWDED_THRESHOLD
    ]
    if len(pickups) >= 2 and len(all_receps) >= 2:
        for _ in range(min(15, len(pickups))):
            idx = rng.choice(len(pickups), 2, replace=False)
            o1, o2 = pickups[idx[0]], pickups[idx[1]]

            compat1 = COMPAT_TABLE.get(o1["objectType"], [])
            compat2 = COMPAT_TABLE.get(o2["objectType"], [])
            p1 = sq.parent_of(o1)
            p2 = sq.parent_of(o2)
            dests1 = [
                r
                for r in all_receps
                if r["objectType"] in compat1
                and (not p1 or r["objectId"] != p1["objectId"])
            ]
            dests2 = [
                r
                for r in all_receps
                if r["objectType"] in compat2
                and (not p2 or r["objectId"] != p2["objectId"])
            ]
            if not dests1 or not dests2:
                continue

            d1 = dests1[rng.randint(len(dests1))]
            d2 = dests2[rng.randint(len(dests2))]
            # 尽量让两个物体放到不同容器
            if d1["objectId"] == d2["objectId"] and len(dests2) > 1:
                d2 = dests2[(rng.randint(len(dests2) - 1) + 1) % len(dests2)]

            task = make_l1_pick_two_and_place(sq, o1, o2, d1, d2)
            if task:
                candidates.append(task)

    return candidates
