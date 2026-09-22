"""从场景元数据中查找物体的工具函数。"""

import json
import os
import random
from typing import Optional

from ...scene_utils import find_objects_by_type
from ...experience_templates import PropertyFilter

# pick_up_and_put.json: [{ObjectType: [receptacle1, receptacle2, ...]}, ...]
# 缓存避免重复读取
_PICK_UP_PUT_PATH = os.path.join(
    os.path.dirname(__file__), "../../assets/pick_up_and_put.json"
)
_valid_receptacles_cache: dict[str, list[str]] = {}


def _get_valid_receptacles(object_type: str) -> list[str]:
    """从 pick_up_and_put.json 获取指定物体类型可放置的容器类型列表。"""
    if not _valid_receptacles_cache:
        try:
            with open(_PICK_UP_PUT_PATH) as f:
                for entry in json.load(f):
                    for obj_t, receptacles in entry.items():
                        _valid_receptacles_cache[obj_t] = receptacles
        except Exception:
            pass
    return _valid_receptacles_cache.get(object_type, [])


# 在有效容器中优先选择这些类型（稳定、常见、语义合理）
_PREFERRED_RECEPTACLE_ORDER = [
    "CounterTop",
    "Shelf",
    "DiningTable",
    "Cabinet",
    "SideTable",
    "CoffeeTable",
    "Desk",
    "TVStand",
    "Dresser",
    "Drawer",
    "Box",
    "Bed",
    "ArmChair",
    "Sofa",
    "Ottoman",
]


def find_objects_by_filter(meta: dict, pf: Optional[PropertyFilter]) -> list[dict]:
    """
    查找满足过滤器条件的所有物体。

    支持两种匹配模式：
      - property_key="salientMaterials": 子串匹配（支持 "|" 分隔多个候选）
      - property_key="objectType":       完全匹配
    """
    if pf is None:
        return []
    results = []
    for obj in meta["objects"]:
        if pf.property_key == "salientMaterials":
            mats = obj.get("salientMaterials") or []
            tokens = [t.strip().lower() for t in pf.value.split("|") if t.strip()] or [
                pf.value.lower()
            ]
            # 要求主材质（index 0）匹配，防止 AluminumFoil 等"次要含纸"的物体被误匹配
            if mats and any(tok == mats[0].lower() for tok in tokens):
                results.append(obj)
        elif pf.property_key == "objectType":
            tokens = [t.strip() for t in pf.value.split("|") if t.strip()] or [pf.value]
            if obj.get("objectType") in tokens:
                results.append(obj)
        elif pf.property_key == "objectType_prefix":
            if obj.get("objectType", "").startswith(pf.value):
                results.append(obj)
    return results


def find_one_object_by_filter(meta: dict, pf: PropertyFilter) -> Optional[dict]:
    """查找一个满足条件的物体，未找到返回 None。"""
    objs = find_objects_by_filter(meta, pf)
    return objs[0] if objs else None


def find_wrong_placement_target(meta: dict, preferred_type: str) -> Optional[dict]:
    """为 owner_habit 规则找一个"错误放置点"。

    从场景 metadata 中随机选一个可以容纳物品的容器，但不是 preferred_type 指定的正确位置。
    """
    candidates = [
        obj
        for obj in meta["objects"]
        if obj.get("receptacle") and obj.get("objectType") != preferred_type
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def find_all_safe_containers(
    meta: dict, failure_object_type: str = "", avoid_object_id: str = ""
) -> list[dict]:
    """返回所有合法安全容器，按 _PREFERRED_RECEPTACLE_ORDER 优先顺序排列（用于自动重试）。"""
    valid = _get_valid_receptacles(failure_object_type)
    search_order = (
        (
            [t for t in _PREFERRED_RECEPTACLE_ORDER if t in valid]
            + [t for t in valid if t not in _PREFERRED_RECEPTACLE_ORDER]
        )
        if valid
        else _PREFERRED_RECEPTACLE_ORDER
    )

    result = []
    seen_ids: set[str] = {avoid_object_id}
    for rtype in search_order:
        for obj in meta["objects"]:
            if (
                obj["objectType"] == rtype
                and obj.get("receptacle")
                and obj["objectId"] not in seen_ids
            ):
                result.append(obj)
                seen_ids.add(obj["objectId"])
    return result


def find_safe_container(
    meta: dict, failure_object_type: str = "", avoid_object_id: str = ""
) -> Optional[dict]:
    """根据 pick_up_and_put.json 规则，为 failure_object_type 找一个合法的放置容器。

    按 _PREFERRED_RECEPTACLE_ORDER 优先顺序筛选；若规则中没有该物体则回落到任意可用容器。
    """
    valid = _get_valid_receptacles(failure_object_type)

    # 按偏好顺序遍历，取第一个既在合法列表中又存在于场景的容器
    search_order = (
        (
            [t for t in _PREFERRED_RECEPTACLE_ORDER if t in valid]
            + [t for t in valid if t not in _PREFERRED_RECEPTACLE_ORDER]
        )
        if valid
        else _PREFERRED_RECEPTACLE_ORDER
    )

    for rtype in search_order:
        for obj in meta["objects"]:
            if (
                obj["objectType"] == rtype
                and obj.get("receptacle")
                and obj["objectId"] != avoid_object_id
            ):
                return obj

    # 最终回落：任意可用容器
    for obj in meta["objects"]:
        if obj.get("receptacle") and obj["objectId"] != avoid_object_id:
            return obj
    return None
