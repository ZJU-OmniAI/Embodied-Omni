"""场景元数据查询封装（从 task_builder.py 抽出）"""

from typing import Optional


class SceneQuery:
    """对 AI2-THOR 场景元数据的高级查询接口"""

    def __init__(self, metadata: dict):
        self.objects = metadata["objects"]
        self._by_id = {o["objectId"]: o for o in self.objects}
        self._by_type: dict[str, list[dict]] = {}
        for o in self.objects:
            self._by_type.setdefault(o["objectType"], []).append(o)

    def get(self, object_id: str) -> Optional[dict]:
        return self._by_id.get(object_id)

    def by_type(self, obj_type: str) -> list[dict]:
        return self._by_type.get(obj_type, [])

    def pickupables(self) -> list[dict]:
        return [
            o for o in self.objects if o.get("pickupable") and not o.get("isPickedUp")
        ]

    def receptacles(self, exclude_floor=True) -> list[dict]:
        r = [o for o in self.objects if o.get("receptacle")]
        if exclude_floor:
            r = [o for o in r if o["objectType"] != "Floor"]
        return r

    def receptacles_by_type(self) -> dict[str, list[dict]]:
        result: dict[str, list[dict]] = {}
        for o in self.receptacles():
            result.setdefault(o["objectType"], []).append(o)
        return result

    def parent_of(self, obj: dict) -> Optional[dict]:
        parents = obj.get("parentReceptacles") or []
        return self.get(parents[0]) if parents else None

    def is_in_closed_container(self, obj: dict) -> bool:
        parent = self.parent_of(obj)
        return (
            parent is not None
            and parent.get("openable")
            and not parent.get("isOpen", False)
        )

    def recep_load(self) -> dict[str, int]:
        """统计每个容器当前包含的物体数量"""
        load: dict[str, int] = {}
        for o in self.objects:
            for pid in o.get("parentReceptacles") or []:
                load[pid] = load.get(pid, 0) + 1
        return load
