"""场景元数据加载与查询工具

AI2-THOR 的场景以 FloorPlan 编号区分房间类型：
  1–30    Kitchen（厨房）
  201–230 LivingRoom（客厅）
  301–330 Bedroom（卧室）
  401–430 Bathroom（浴室）

元数据文件预先从 AI2-THOR 模拟器导出，存放于
emem_bench.construction/assets/scene_metadata/<room_folder>/<scene>/metadata.json
离线使用，不需要启动模拟器。
"""

import os
import json

from .procthor import (
    is_procthor_scene,
    load_procthor_metadata,
    parse_procthor_scene_ref,
    procthor_primary_room_type,
)

# 元数据根目录，与本文件同级的 assets/ 下
METADATA_ROOT = os.environ.get(
    "EMEM_SCENE_METADATA_ROOT",
    os.path.join(
        os.environ.get("EMEM_DATA_ROOT", "data"),
        "data_engine",
        "assets",
        "scene_metadata",
    ),
)

# FloorPlan 房间类型 → 子目录名的映射
ROOM_TYPE_MAP = {
    "Kitchen": "kitchens",
    "LivingRoom": "living_rooms",
    "Bedroom": "bedrooms",
    "Bathroom": "bathrooms",
}


def procthor_scene_metadata_dir(scene: str) -> str:
    """返回 ProcTHOR 离线 metadata 目录。

    目录结构与 AI2-THOR 场景保持一致，每个场景一个目录，下面包含
    metadata.json 和 originPos.json：
      emem_bench.construction/assets/scene_metadata/procthor/test/ProcTHOR-test-000000/
    """
    split, _ = parse_procthor_scene_ref(scene)
    return os.path.join(METADATA_ROOT, "procthor", split, scene)


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[0] if isinstance(data, list) else data


def get_room_type(scene: str | dict) -> str:
    """根据场景名推断房间类型。

    Args:
        scene: 场景名，如 "FloorPlan1"、"FloorPlan201"

    Returns:
        "Kitchen" / "LivingRoom" / "Bedroom" / "Bathroom"

    Raises:
        ValueError: 场景编号不在已知范围内
    """
    if isinstance(scene, dict) or is_procthor_scene(scene):
        return procthor_primary_room_type(scene)

    num = int(str(scene).replace("FloorPlan", ""))
    if 1 <= num <= 30:
        return "Kitchen"
    elif 201 <= num <= 230:
        return "LivingRoom"
    elif 301 <= num <= 330:
        return "Bedroom"
    elif 401 <= num <= 430:
        return "Bathroom"
    raise ValueError(f"Unknown scene: {scene}")


def load_scene_metadata(scene: str | dict) -> dict:
    """加载场景的 AI2-THOR 元数据（离线，不启动模拟器）。

    元数据包含场景内所有物体的状态快照：位置、类型、可交互属性
    （pickupable、receptacle、openable 等）、材质、父容器等。

    Args:
        scene: 场景名，如 "FloorPlan1"

    Returns:
        metadata dict，结构与 controller.last_event.metadata 一致，
        顶层键包括 "objects"（物体列表）和 "agent"（智能体初始状态）
    """
    if isinstance(scene, str) and is_procthor_scene(scene):
        meta_path = os.path.join(procthor_scene_metadata_dir(scene), "metadata.json")
        if os.path.exists(meta_path):
            return _load_json(meta_path)
        return load_procthor_metadata(scene)

    if isinstance(scene, dict):
        return load_procthor_metadata(scene if isinstance(scene, str) else scene)

    room_type = get_room_type(scene)
    folder = ROOM_TYPE_MAP[room_type]
    meta_path = os.path.join(METADATA_ROOT, folder, scene, "metadata.json")
    return _load_json(meta_path)


def load_agent_origin(scene: str | dict) -> dict:
    """加载智能体在该场景的初始出生点坐标。

    Returns:
        dict，包含 position（x/y/z）和 rotation 字段
    """
    if isinstance(scene, str) and is_procthor_scene(scene):
        origin_path = os.path.join(procthor_scene_metadata_dir(scene), "originPos.json")
        if os.path.exists(origin_path):
            return _load_json(origin_path)
        meta = load_procthor_metadata(scene if isinstance(scene, str) else scene)
        agent = meta.get("agent") or {}
        return {
            "position": agent.get("position", {}),
            "rotation": agent.get("rotation", {}),
            "standing": agent.get("standing", True),
            "horizon": agent.get("horizon", 30),
        }

    if isinstance(scene, dict):
        meta = load_procthor_metadata(scene)
        agent = meta.get("agent") or {}
        return {
            "position": agent.get("position", {}),
            "rotation": agent.get("rotation", {}),
            "standing": agent.get("standing", True),
            "horizon": agent.get("horizon", 30),
        }

    room_type = get_room_type(scene)
    folder = ROOM_TYPE_MAP[room_type]
    pos_path = os.path.join(METADATA_ROOT, folder, scene, "originPos.json")
    return _load_json(pos_path)


def find_objects_by_type(metadata: dict, obj_type: str) -> list[dict]:
    """返回场景中所有指定类型的物体。

    Args:
        obj_type: AI2-THOR 物体类型名，如 "Apple"、"SinkBasin"
    """
    return [obj for obj in metadata["objects"] if obj["objectType"] == obj_type]


def find_objects_by_property(metadata: dict, prop: str, value=True) -> list[dict]:
    """返回场景中指定属性等于给定值的所有物体。

    常用属性：
        "pickupable"  — 可拾取
        "receptacle"  — 可作为容器
        "openable"    — 可开关
        "isToggled"   — 当前已打开（如水龙头、灯）

    Args:
        prop:  物体属性键名
        value: 目标值，默认为 True
    """
    return [obj for obj in metadata["objects"] if obj.get(prop) == value]
