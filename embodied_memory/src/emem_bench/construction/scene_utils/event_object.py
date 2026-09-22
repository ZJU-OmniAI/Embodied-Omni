import ai2thor.server
from typing import List, Dict, Tuple


class EventObject:
    def __init__(self, event: ai2thor.server.Event):
        self.objects = event.metadata["objects"]
        self.object2color = event.object_id_to_color
        self.color2object = event.color_to_object_id
        _, self.item2object = self.get_objects()

    def get_objects(self) -> Tuple[List[dict], Dict[str, dict]]:
        item2object = {}
        for item in self.objects:
            item2object[item["name"]] = item
        return self.objects, item2object

    def get_all_item_position(self) -> dict:
        return {item["name"]: item["position"] for item in self.objects}

    def get_visible_objects(self) -> Tuple[List[dict], List[dict]]:
        return (
            [obj["name"] for obj in self.objects if obj["visible"]],
            [obj for obj in self.objects if obj["visible"]],
        )

    def get_isInteractable_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["isInteractable"]]

    def get_receptacle_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["receptacle"]]

    def get_toggleable_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["toggleable"]]

    def get_breakable_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["breakable"]]

    def get_isToggled_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["isToggled"]]

    def get_isBroken_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["isBroken"]]

    def get_canFillWithLiquid_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["canFillWithLiquid"]]

    def get_isFilledWithLiquid_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["isFilledWithLiquid"]]

    def get_fillLiquid_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["fillLiquid"]]

    def get_dirtyable_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["dirtyable"]]

    def get_isDirty_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["isDirty"]]

    def get_canBeUsedUp_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["canBeUsedUp"]]

    def get_isUsedUp_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["isUsedUp"]]

    def get_cookable_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["cookable"]]

    def get_isCooked_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["isCooked"]]

    def get_isHeatSource_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["isHeatSource"]]

    def get_isColdSource_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["isColdSource"]]

    def get_sliceable_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["sliceable"]]

    def get_openable_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["openable"]]

    def get_isOpen_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["isOpen"]]

    def get_pickupable_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["pickupable"]]

    def get_isPickedUp_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["isPickedUp"]]

    def get_moveable_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["moveable"]]

    def get_isMoving_objects(self) -> List[dict]:
        return [obj for obj in self.objects if obj["isMoving"]]

    def get_object_color(self, object_id: str) -> str:
        return self.object2color[object_id]

    def get_color_object(self, color: str):
        return self.color2object[color]

    def get_item_mass(self, item_name: str) -> float:
        return self.item2object[item_name]["mass"]

    def get_item_volume(self, item_name: str) -> float:
        size = self.item2object[item_name]["axisAlignedBoundingBox"]["size"]
        return round(size["x"] * size["y"] * size["z"], 4)

    def get_item_surface_area(self, item_name: str) -> float:
        size = self.item2object[item_name]["axisAlignedBoundingBox"]["size"]
        x, y, z = size["x"], size["y"], size["z"]
        return round(max(x * y, x * z, y * z), 4)

    def get_item_position(self, item_name: str) -> dict:
        return self.item2object[item_name]["position"]

    def get_item_orientation(self, item_name: str) -> dict:
        return self.item2object[item_name]["rotation"]
