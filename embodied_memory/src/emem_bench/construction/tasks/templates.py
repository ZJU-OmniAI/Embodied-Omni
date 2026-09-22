"""L0/L1 任务模板函数及物体属性常量

每个 make_xxx 函数接受场景查询对象和物体 dict，
返回一个 Task（失败则返回 None）。
"""

from __future__ import annotations
import json
import os
from typing import Optional

from ..scene_utils import SceneQuery
from .schema import ActionStep, Task

# ─── 物体-容器兼容表 ──────────────────────────────────────────
# 来源: assets/pick_up_and_put.json，记录每种物体可以放入的容器类型

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
_COMPAT_PATH = os.path.join(_ASSETS_DIR, "pick_up_and_put.json")


def _load_compatibility() -> dict[str, list[str]]:
    with open(_COMPAT_PATH, "r") as f:
        raw = json.load(f)
    table = {}
    for entry in raw:
        for obj_type, receps in entry.items():
            table[obj_type.strip()] = [r.strip() for r in receps if r.strip()]
    return table


COMPAT_TABLE = _load_compatibility()

# AI2-THOR 中可进行状态变换的物体类型
HEATABLE = {"Apple", "Bread", "Cup", "Egg", "Mug", "Plate", "Potato", "Tomato"}
COOLABLE = {
    "Apple",
    "Bowl",
    "Bread",
    "Cup",
    "Egg",
    "Lettuce",
    "Mug",
    "Pan",
    "Plate",
    "Potato",
    "Tomato",
    "WineBottle",
}
CLEANABLE = {
    "Apple",
    "Bowl",
    "ButterKnife",
    "Cloth",
    "Cup",
    "DishSponge",
    "Egg",
    "Fork",
    "Kettle",
    "Knife",
    "Ladle",
    "Lettuce",
    "Mug",
    "Pan",
    "Plate",
    "Pot",
    "Potato",
    "SoapBar",
    "Spatula",
    "Spoon",
    "Tomato",
}


# ─── 通用步骤片段 ─────────────────────────────────────────────


def _make_pickup_steps(sq: SceneQuery, obj: dict) -> list[ActionStep]:
    """生成拿起物体的步骤，自动处理关闭容器（先开门再拿，拿完关门）"""
    obj_type = obj["objectType"]
    parent = sq.parent_of(obj)
    if not parent:
        return []

    if sq.is_in_closed_container(obj):
        return [
            ActionStep(
                "Navigate",
                parent["objectId"],
                parent["objectType"],
                f"Navigate to {parent['objectType']}",
            ),
            ActionStep(
                "Open",
                parent["objectId"],
                parent["objectType"],
                f"Open {parent['objectType']}",
            ),
            ActionStep("PickUp", obj["objectId"], obj_type, f"Pick up {obj_type}"),
            ActionStep(
                "Close",
                parent["objectId"],
                parent["objectType"],
                f"Close {parent['objectType']}",
            ),
        ]
    else:
        return [
            ActionStep(
                "Navigate",
                parent["objectId"],
                parent["objectType"],
                f"Navigate to {parent['objectType']}",
            ),
            ActionStep("PickUp", obj["objectId"], obj_type, f"Pick up {obj_type}"),
        ]


def _make_place_steps(dest: dict, obj_type: str, prefix: str = "") -> list[ActionStep]:
    """生成放置步骤，自动处理 openable 容器（冰箱/柜子等：先开门再放，放完关门）"""
    dest_type = dest["objectType"]
    label = f"{prefix}{obj_type}" if prefix else obj_type

    if dest.get("openable") and not dest.get("isOpen", False):
        return [
            ActionStep(
                "Navigate", dest["objectId"], dest_type, f"Navigate to {dest_type}"
            ),
            ActionStep("Open", dest["objectId"], dest_type, f"Open {dest_type}"),
            ActionStep(
                "PutObject",
                dest["objectId"],
                dest_type,
                f"Put {label} into {dest_type}",
            ),
            ActionStep("Close", dest["objectId"], dest_type, f"Close {dest_type}"),
        ]
    else:
        return [
            ActionStep(
                "Navigate", dest["objectId"], dest_type, f"Navigate to {dest_type}"
            ),
            ActionStep(
                "PutObject", dest["objectId"], dest_type, f"Put {label} on {dest_type}"
            ),
        ]


# ─── L0 模板 ─────────────────────────────────────────────────


def make_l0_pick_and_place(sq: SceneQuery, obj: dict, dest: dict) -> Optional[Task]:
    """L0: 简单搬运（< 10 步）Navigate → PickUp → Navigate → PutObject"""
    obj_type = obj["objectType"]
    dest_type = dest["objectType"]
    if dest_type not in COMPAT_TABLE.get(obj_type, []):
        return None
    if not sq.parent_of(obj):
        return None

    steps = _make_pickup_steps(sq, obj) + _make_place_steps(dest, obj_type)
    name = (
        f"Put {obj_type} into {dest_type}"
        if dest.get("openable")
        else f"Put {obj_type} on {dest_type}"
    )
    return Task(
        name=name,
        level="L0",
        task_type="pick_and_place",
        steps=steps,
        key_object=obj["objectId"],
    )


# ─── L1 模板 ─────────────────────────────────────────────────


def make_l1_clean_and_place(sq: SceneQuery, obj: dict, dest: dict) -> Optional[Task]:
    """L1: 清洗后放置（约 13 步）需要工作记忆：记住多步计划和最终目标位置"""
    obj_type = obj["objectType"]
    if obj_type not in CLEANABLE:
        return None
    if dest["objectType"] not in COMPAT_TABLE.get(obj_type, []):
        return None
    if not sq.parent_of(obj):
        return None

    sinks = sq.by_type("SinkBasin")
    faucets = sq.by_type("Faucet")
    if not sinks or not faucets:
        return None
    sink, faucet = sinks[0], faucets[0]

    dest_label = "into" if dest.get("openable") else "on"
    steps = (
        _make_pickup_steps(sq, obj)
        + [
            ActionStep(
                "Navigate", sink["objectId"], "SinkBasin", "Navigate to SinkBasin"
            ),
            ActionStep(
                "PutObject",
                sink["objectId"],
                "SinkBasin",
                f"Put {obj_type} into SinkBasin",
            ),
            ActionStep("Navigate", faucet["objectId"], "Faucet", "Navigate to Faucet"),
            ActionStep(
                "ToggleOn", faucet["objectId"], "Faucet", "Turn on Faucet to rinse"
            ),
            ActionStep("ToggleOff", faucet["objectId"], "Faucet", "Turn off Faucet"),
            ActionStep(
                "Navigate", sink["objectId"], "SinkBasin", "Return to SinkBasin"
            ),
            ActionStep(
                "PickUp", obj["objectId"], obj_type, f"Pick up cleaned {obj_type}"
            ),
        ]
        + _make_place_steps(dest, f"cleaned {obj_type}")
    )

    return Task(
        name=f"Clean {obj_type} and place it {dest_label} {dest['objectType']}",
        level="L1",
        task_type="clean_and_place",
        steps=steps,
        key_object=obj["objectId"],
    )


def make_l1_heat_and_place(sq: SceneQuery, obj: dict, dest: dict) -> Optional[Task]:
    """L1: 加热后放置（约 13 步）需要工作记忆：记住开关序列和最终目标"""
    obj_type = obj["objectType"]
    if obj_type not in HEATABLE:
        return None
    if dest["objectType"] not in COMPAT_TABLE.get(obj_type, []):
        return None
    if dest["objectType"] == "Microwave":
        return None
    if not sq.parent_of(obj):
        return None

    microwaves = sq.by_type("Microwave")
    if not microwaves:
        return None
    mw = microwaves[0]

    dest_label = "into" if dest.get("openable") else "on"
    steps = (
        _make_pickup_steps(sq, obj)
        + [
            ActionStep(
                "Navigate", mw["objectId"], "Microwave", "Navigate to Microwave"
            ),
            ActionStep("Open", mw["objectId"], "Microwave", "Open Microwave door"),
            ActionStep(
                "PutObject",
                mw["objectId"],
                "Microwave",
                f"Put {obj_type} into Microwave",
            ),
            ActionStep("Close", mw["objectId"], "Microwave", "Close Microwave door"),
            ActionStep(
                "ToggleOn", mw["objectId"], "Microwave", "Turn on Microwave to heat"
            ),
            ActionStep("ToggleOff", mw["objectId"], "Microwave", "Turn off Microwave"),
            ActionStep(
                "Open",
                mw["objectId"],
                "Microwave",
                "Open Microwave door to retrieve item",
            ),
            ActionStep(
                "PickUp", obj["objectId"], obj_type, f"Pick up heated {obj_type}"
            ),
            ActionStep("Close", mw["objectId"], "Microwave", "Close Microwave door"),
        ]
        + _make_place_steps(dest, f"heated {obj_type}")
    )

    return Task(
        name=f"Heat {obj_type} and place it {dest_label} {dest['objectType']}",
        level="L1",
        task_type="heat_and_place",
        steps=steps,
        key_object=obj["objectId"],
    )


def make_l1_cool_and_place(sq: SceneQuery, obj: dict, dest: dict) -> Optional[Task]:
    """L1: 冷藏后放置（约 11 步）需要工作记忆：记住冰箱操作序列和最终目标"""
    obj_type = obj["objectType"]
    if obj_type not in COOLABLE:
        return None
    if dest["objectType"] not in COMPAT_TABLE.get(obj_type, []):
        return None
    if dest["objectType"] == "Fridge":
        return None
    if not sq.parent_of(obj):
        return None

    fridges = sq.by_type("Fridge")
    if not fridges:
        return None
    fridge = fridges[0]

    dest_label = "into" if dest.get("openable") else "on"
    steps = (
        _make_pickup_steps(sq, obj)
        + [
            ActionStep("Navigate", fridge["objectId"], "Fridge", "Navigate to Fridge"),
            ActionStep("Open", fridge["objectId"], "Fridge", "Open Fridge"),
            ActionStep(
                "PutObject", fridge["objectId"], "Fridge", f"Put {obj_type} into Fridge"
            ),
            ActionStep(
                "Close", fridge["objectId"], "Fridge", "Close Fridge to cool item"
            ),
            ActionStep(
                "Open", fridge["objectId"], "Fridge", "Open Fridge to retrieve item"
            ),
            ActionStep(
                "PickUp", obj["objectId"], obj_type, f"Pick up cooled {obj_type}"
            ),
            ActionStep("Close", fridge["objectId"], "Fridge", "Close Fridge"),
        ]
        + _make_place_steps(dest, f"cooled {obj_type}")
    )

    return Task(
        name=f"Cool {obj_type} and place it {dest_label} {dest['objectType']}",
        level="L1",
        task_type="cool_and_place",
        steps=steps,
        key_object=obj["objectId"],
    )


def make_l1_pick_two_and_place(
    sq: SceneQuery,
    obj1: dict,
    obj2: dict,
    dest1: dict,
    dest2: dict,
) -> Optional[Task]:
    """L1: 搬两个物体到各自目标（约 8-16 步）需要工作记忆：记住两个物体和两个目标"""
    t1, t2 = obj1["objectType"], obj2["objectType"]
    if dest1["objectType"] not in COMPAT_TABLE.get(t1, []):
        return None
    if dest2["objectType"] not in COMPAT_TABLE.get(t2, []):
        return None
    if not sq.parent_of(obj1) or not sq.parent_of(obj2):
        return None

    steps = (
        _make_pickup_steps(sq, obj1)
        + _make_place_steps(dest1, t1)
        + _make_pickup_steps(sq, obj2)
        + _make_place_steps(dest2, t2)
    )

    return Task(
        name=f"Put {t1} on {dest1['objectType']}, then put {t2} on {dest2['objectType']}",
        level="L1",
        task_type="pick_two_and_place",
        steps=steps,
        key_object=obj1["objectId"],
    )
