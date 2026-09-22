"""ProcTHOR scene references and lightweight loading helpers.

Generated episodes should store compact, stable scene strings instead of
embedding full ProcTHOR house dictionaries.  A reference has the form:

    ProcTHOR-test-000123

The helper functions below resolve that reference to the house dict only when a
Controller or metadata snapshot is needed.
"""

from __future__ import annotations

import copy
import re
from functools import lru_cache
from typing import Any


_REF_RE = re.compile(r"^ProcTHOR-(?P<split>train|val|test)-(?P<index>\d+)$")
_VALID_SPLITS = {"train", "val", "test"}
_ALL_SPLITS = ("train", "val", "test")


def make_procthor_scene_ref(split: str, index: int) -> str:
    if split not in _VALID_SPLITS:
        raise ValueError(f"Unsupported ProcTHOR split: {split}")
    if index < 0:
        raise ValueError("ProcTHOR index must be non-negative")
    return f"ProcTHOR-{split}-{index:06d}"


def is_procthor_scene(scene: Any) -> bool:
    return isinstance(scene, dict) or (
        isinstance(scene, str) and _REF_RE.match(scene) is not None
    )


def parse_procthor_scene_ref(scene: str) -> tuple[str, int]:
    match = _REF_RE.match(scene)
    if not match:
        raise ValueError(f"Not a ProcTHOR scene reference: {scene}")
    return match.group("split"), int(match.group("index"))


@lru_cache(maxsize=1)
def load_procthor_dataset():
    import prior

    return prior.load_dataset("procthor-10k", offline=True)


def procthor_split_size(split: str) -> int:
    if split == "all":
        return sum(len(getattr(load_procthor_dataset(), name)) for name in _ALL_SPLITS)
    if split not in _VALID_SPLITS:
        raise ValueError(f"Unsupported ProcTHOR split: {split}")
    return len(getattr(load_procthor_dataset(), split))


@lru_cache(maxsize=2048)
def load_procthor_house(split: str, index: int) -> dict:
    if split not in _VALID_SPLITS:
        raise ValueError(f"Unsupported ProcTHOR split: {split}")
    data = getattr(load_procthor_dataset(), split)
    if index < 0 or index >= len(data):
        raise IndexError(f"ProcTHOR {split} index out of range: {index}")
    return copy.deepcopy(data[index])


def resolve_controller_scene(scene: str | dict) -> str | dict:
    if isinstance(scene, dict):
        return scene
    if isinstance(scene, str) and _REF_RE.match(scene):
        split, index = parse_procthor_scene_ref(scene)
        return load_procthor_house(split, index)
    return scene


def scene_ref_from_house_metadata(house: dict) -> str:
    metadata = house.get("metadata") or {}
    source = metadata.get("sourceScene") or metadata.get("sceneName")
    return str(source or "ProcTHORHouse")


def procthor_primary_room_type(scene: str | dict) -> str:
    """Return a readable room label for ProcTHOR scenes.

    ProcTHOR houses are often multi-room.  For benchmark text we prefer the
    agent's starting room if it can be inferred; otherwise use the first room
    type or a generic label.
    """
    house = scene if isinstance(scene, dict) else resolve_controller_scene(scene)
    if not isinstance(house, dict):
        return "ProcTHORHouse"

    rooms = house.get("rooms") or []
    if not rooms:
        return "ProcTHORHouse"

    agent_pos = ((house.get("metadata") or {}).get("agent") or {}).get("position")
    if agent_pos:
        for room in rooms:
            if _point_in_polygon(
                agent_pos.get("x"),
                agent_pos.get("z"),
                [(p.get("x"), p.get("z")) for p in room.get("floorPolygon") or []],
            ):
                return room.get("roomType") or "ProcTHORHouse"

    return rooms[0].get("roomType") or "ProcTHORHouse"


def _point_in_polygon(x: float | None, z: float | None, polygon: list[tuple]) -> bool:
    if x is None or z is None or len(polygon) < 3:
        return False
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, zi = polygon[i]
        xj, zj = polygon[j]
        if xi is None or zi is None or xj is None or zj is None:
            j = i
            continue
        intersects = ((zi > z) != (zj > z)) and (
            x < (xj - xi) * (z - zi) / ((zj - zi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _controller_metadata(scene_input: str | dict) -> dict:
    from ai2thor.controller import Controller

    platform = None
    try:
        from ai2thor.platform import CloudRendering

        platform = CloudRendering
    except ImportError:
        platform = None

    kwargs = dict(
        agentMode="default",
        visibilityDistance=1.5,
        scene=resolve_controller_scene(scene_input),
        gridSize=0.25,
        snapToGrid=True,
        rotateStepDegrees=90,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width=300,
        height=300,
        fieldOfView=90,
    )
    if platform:
        kwargs["platform"] = platform
    controller = Controller(**kwargs)
    try:
        return copy.deepcopy(controller.last_event.metadata)
    finally:
        controller.stop()


@lru_cache(maxsize=128)
def _load_procthor_metadata_cached(scene: str) -> dict:
    return _controller_metadata(scene)


def load_procthor_metadata(scene: str | dict) -> dict:
    if isinstance(scene, dict):
        return _controller_metadata(scene)
    return copy.deepcopy(_load_procthor_metadata_cached(scene))


def list_procthor_scene_refs(
    split: str = "test", limit: int | None = None, start: int = 0
) -> list[str]:
    if start < 0:
        raise ValueError("ProcTHOR start must be non-negative")
    if split == "all":
        refs: list[str] = []
        for name in _ALL_SPLITS:
            refs.extend(list_procthor_scene_refs(name))
        end = len(refs) if limit is None else min(len(refs), start + limit)
        return refs[start:end]

    size = procthor_split_size(split)
    if start >= size:
        return []
    end = size if limit is None else min(size, start + limit)
    return [make_procthor_scene_ref(split, idx) for idx in range(start, end)]


def list_procthor_scene_refs_all(limit: int | None = None, start: int = 0) -> list[str]:
    return list_procthor_scene_refs("all", limit=limit, start=start)
