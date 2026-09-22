from .event_object import EventObject
from .base_action import BaseAction
import math
import time
from PIL import Image
import numpy as np
from abc import ABC


class BaseAgent(ABC):
    def __init__(self, controller):
        self.controller = controller
        self.eventobject = EventObject(self.controller.last_event)
        self.step_count = 0
        self.last_action = "INIT"
        self.mermory = []
        self.action = BaseAction()

    def update_event(self):
        self.eventobject = EventObject(self.controller.last_event)
        self.controller.step(action="Pass")

    def get_agent_position(self):
        return self.controller.last_event.metadata["agent"]["position"]

    def get_agent_rotation(self):
        return self.controller.last_event.metadata["agent"]["rotation"]

    def get_agent_horizon(self):
        return self.controller.last_event.metadata["agent"]["cameraHorizon"]

    def get_camera_position(self):
        return self.controller.last_event.metadata["cameraPosition"]

    def get_camera_rotation(self):
        return self.controller.last_event.pose_discrete[3]

    def compute_position(self, item):
        target_position = None
        target_rotation = None
        event = self.controller.step(
            dict(action="GetInteractablePoses", objectId=item["objectId"])
        )
        reachable_positions = event.metadata.get("actionReturn") or []
        if len(reachable_positions) == 0:
            print("No reachable positions found.")
            return target_position, target_rotation
        front_positions = []
        side_positions = []

        for position in reachable_positions:
            if round(abs(position["rotation"] - item["rotation"]["y"])) == 180:
                front_positions.append(position)
            elif round(abs(position["rotation"] - item["rotation"]["y"])) == 90:
                side_positions.append(position)

        if len(front_positions) > 0:
            max_distance = 0
            for position in front_positions:
                distance = math.sqrt(
                    (position["x"] - item["position"]["x"]) ** 2
                    + (position["z"] - item["position"]["z"]) ** 2
                )
                if distance > max_distance:
                    max_distance = distance
                    target_position = position

        if target_position is None and len(side_positions) > 0:
            max_distance = 0
            for position in side_positions:
                distance = math.sqrt(
                    (position["x"] - item["position"]["x"]) ** 2
                    + (position["z"] - item["position"]["z"]) ** 2
                )
                if distance > max_distance:
                    max_distance = distance
                    target_position = position

        if target_position is None:
            max_distance = 0
            for position in reachable_positions:
                distance = math.sqrt(
                    (position["x"] - item["position"]["x"]) ** 2
                    + (position["z"] - item["position"]["z"]) ** 2
                )
                if distance > max_distance:
                    max_distance = distance
                    target_position = position

        return target_position, dict(x=0, y=target_position["rotation"], z=0)

    def compute_position_1(self, item, reachable_positions):
        target_position = None
        min_distance = float("inf")
        for position in reachable_positions:
            distance = math.sqrt(
                (position["x"] - item["position"]["x"]) ** 2
                + (position["z"] - item["position"]["z"]) ** 2
            )
            if distance < min_distance:
                min_distance = distance
                target_position = position
        return target_position, dict(x=0, y=target_position["rotation"], z=0)

    def compute_position_8(self, item, pre_target_positions):
        target_position = None
        target_rotation = None
        event = self.controller.step(
            dict(action="GetInteractablePoses", objectId=item["objectId"])
        )
        reachable_positions = event.metadata.get("actionReturn") or []
        reachable_positions = [
            position
            for position in reachable_positions
            if math.sqrt(
                (position["x"] - item["position"]["x"]) ** 2
                + (position["z"] - item["position"]["z"]) ** 2
            )
            <= 1.5
        ]
        if len(reachable_positions) == 0:
            print("No reachable positions found.")
            return target_position, target_rotation
        if pre_target_positions:
            reachable_positions = [
                p for p in reachable_positions if p not in pre_target_positions
            ]
        if not reachable_positions:
            return None, None

        if (
            self.eventobject.get_item_volume(item["name"]) <= 0.1
            and self.eventobject.get_item_surface_area(item["name"]) <= 1
        ):
            target_position, target_rotation = self.compute_position_1(
                item, reachable_positions
            )
            return target_position, target_rotation

        angles = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        item_rotation = min(
            angles, key=lambda angle: abs(angle - round(item["rotation"]["y"]))
        )
        item_rotation = 0 if item_rotation == 360 else item_rotation
        target_position = None
        target_rotation = None
        candidate_positions = []

        if item_rotation == 180:
            target_rotation = dict(x=0, y=0, z=0)
            for position in reachable_positions:
                if abs(position["x"] - item["position"]["x"]) <= 0.1:
                    candidate_positions.append(position)
            front_positions = [
                p for p in candidate_positions if p["z"] < item["position"]["z"]
            ]
            back_positions = [
                p for p in candidate_positions if p["z"] > item["position"]["z"]
            ]
            if front_positions:
                target_position = self.compute_closest_positions(item, front_positions)
            elif back_positions:
                target_rotation = dict(x=0, y=180, z=0)
                target_position = self.compute_closest_positions(item, back_positions)

        elif item_rotation == 270:
            target_rotation = dict(x=0, y=90, z=0)
            for position in reachable_positions:
                if abs(position["z"] - item["position"]["z"]) <= 0.1:
                    candidate_positions.append(position)
            front_positions = [
                p for p in candidate_positions if p["x"] < item["position"]["x"]
            ]
            back_positions = [
                p for p in candidate_positions if p["x"] > item["position"]["x"]
            ]
            if front_positions:
                target_position = self.compute_closest_positions(item, front_positions)
            elif back_positions:
                target_rotation = dict(x=0, y=270, z=0)
                target_position = self.compute_closest_positions(item, back_positions)

        elif item_rotation == 0:
            target_rotation = dict(x=0, y=180, z=0)
            for position in reachable_positions:
                if abs(position["x"] - item["position"]["x"]) <= 0.1:
                    candidate_positions.append(position)
            for tolerance in [0.2, 0.3, 0.4, 0.5]:
                if not candidate_positions:
                    candidate_positions = [
                        p
                        for p in reachable_positions
                        if abs(p["z"] - item["position"]["z"]) <= tolerance
                    ]
                else:
                    break
            front_positions = [
                p for p in candidate_positions if p["z"] > item["position"]["z"]
            ]
            back_positions = [
                p for p in candidate_positions if p["z"] < item["position"]["z"]
            ]
            target_position_front = (
                self.compute_closest_positions(item, front_positions)
                if front_positions
                else None
            )
            target_position_back = (
                self.compute_closest_positions(item, back_positions)
                if back_positions
                else None
            )
            if target_position_front and target_position_back:
                dist_front = math.sqrt(
                    (target_position_front["x"] - item["position"]["x"]) ** 2
                    + (target_position_front["z"] - item["position"]["z"]) ** 2
                )
                dist_back = math.sqrt(
                    (target_position_back["x"] - item["position"]["x"]) ** 2
                    + (target_position_back["z"] - item["position"]["z"]) ** 2
                )
                if dist_front < dist_back:
                    target_position = target_position_front
                else:
                    target_rotation = dict(x=0, y=0, z=0)
                    target_position = target_position_back
            elif target_position_front:
                target_position = target_position_front
            elif target_position_back:
                target_rotation = dict(x=0, y=0, z=0)
                target_position = target_position_back

        elif item_rotation == 90:
            target_rotation = dict(x=0, y=270, z=0)
            for position in reachable_positions:
                if abs(position["z"] - item["position"]["z"]) <= 0.1:
                    candidate_positions.append(position)
            front_positions = [
                p for p in candidate_positions if p["x"] > item["position"]["x"]
            ]
            back_positions = [
                p for p in candidate_positions if p["x"] < item["position"]["x"]
            ]
            if front_positions:
                target_position = self.compute_closest_positions(item, front_positions)
            elif back_positions:
                target_rotation = dict(x=0, y=90, z=0)
                target_position = self.compute_closest_positions(item, back_positions)

        elif item_rotation == 45:
            target_rotation = dict(x=0, y=225, z=0)
            front_positions = [
                p
                for p in reachable_positions
                if p["x"] > item["position"]["x"] and p["z"] > item["position"]["z"]
            ]
            back_positions = [
                p
                for p in reachable_positions
                if p["x"] < item["position"]["x"] and p["z"] < item["position"]["z"]
            ]
            if front_positions:
                target_position = self.compute_closest_positions(item, front_positions)
            if target_position is None and back_positions:
                target_rotation = dict(x=0, y=45, z=0)
                target_position = self.compute_closest_positions(item, back_positions)

        elif item_rotation == 135:
            target_rotation = dict(x=0, y=315, z=0)
            front_positions = [
                p
                for p in reachable_positions
                if p["x"] > item["position"]["x"] and p["z"] < item["position"]["z"]
            ]
            back_positions = [
                p
                for p in reachable_positions
                if p["x"] < item["position"]["x"] and p["z"] > item["position"]["z"]
            ]
            if front_positions:
                target_position = self.compute_closest_positions(item, front_positions)
            if target_position is None and back_positions:
                target_rotation = dict(x=0, y=135, z=0)
                target_position = self.compute_closest_positions(item, back_positions)

        elif item_rotation == 225:
            target_rotation = dict(x=0, y=45, z=0)
            front_positions = [
                p
                for p in reachable_positions
                if p["x"] < item["position"]["x"] and p["z"] < item["position"]["z"]
            ]
            back_positions = [
                p
                for p in reachable_positions
                if p["x"] > item["position"]["x"] and p["z"] > item["position"]["z"]
            ]
            if front_positions:
                target_position = self.compute_closest_positions(item, front_positions)
            if target_position is None and back_positions:
                target_rotation = dict(x=0, y=225, z=0)
                target_position = self.compute_closest_positions(item, back_positions)

        elif item_rotation == 315:
            target_rotation = dict(x=0, y=135, z=0)
            front_positions = [
                p
                for p in reachable_positions
                if p["x"] < item["position"]["x"] and p["z"] > item["position"]["z"]
            ]
            back_positions = [
                p
                for p in reachable_positions
                if p["x"] > item["position"]["x"] and p["z"] < item["position"]["z"]
            ]
            if front_positions:
                target_position = self.compute_closest_positions(item, front_positions)
            if target_position is None and back_positions:
                target_rotation = dict(x=0, y=315, z=0)
                target_position = self.compute_closest_positions(item, back_positions)

        if target_position is None:
            target_position, target_rotation = self.compute_position_1(
                item, reachable_positions
            )
        return target_position, target_rotation

    def compute_closest_positions(self, item, candidate_positions, gap=0.1):
        item_position = item["position"]
        item_volume = self.eventobject.get_item_volume(item["name"])
        item_surface_area = self.eventobject.get_item_surface_area(item["name"])

        A = 1
        B = -math.tan(math.radians(item["rotation"]["y"]))
        C = -item_position["x"] - B * item_position["z"]

        min_distance = float("inf")
        closest_points = []
        for position in candidate_positions:
            x0, z0 = position["x"], position["z"]
            numerator = abs(A * z0 + B * x0 + C)
            denominator = math.sqrt(A**2 + B**2)
            distance = numerator / denominator
            if distance <= min_distance + gap:
                if distance < min_distance:
                    min_distance = distance
                closest_points.append(position)

        if item_volume <= 0.2 and item_surface_area <= 0.5:
            min_dist = float("inf")
            target_position = None
            for position in closest_points:
                distance = math.sqrt(
                    (position["x"] - item_position["x"]) ** 2
                    + (position["z"] - item_position["z"]) ** 2
                )
                if distance < min_dist:
                    min_dist = distance
                    target_position = position
            return target_position

        elif item_volume <= 1 and item_surface_area <= 1:
            closest_points = sorted(
                closest_points,
                key=lambda p: math.sqrt(
                    (p["x"] - item_position["x"]) ** 2
                    + (p["z"] - item_position["z"]) ** 2
                ),
            )
            return closest_points[len(closest_points) // 2] if closest_points else None
        else:
            max_distance = 0
            target_position = None
            filtered = [
                p
                for p in closest_points
                if math.sqrt(
                    (p["x"] - item_position["x"]) ** 2
                    + (p["z"] - item_position["z"]) ** 2
                )
                <= 1
            ]
            for position in filtered:
                distance = math.sqrt(
                    (position["x"] - item_position["x"]) ** 2
                    + (position["z"] - item_position["z"]) ** 2
                )
                if distance > max_distance:
                    max_distance = distance
                    target_position = position
            return target_position

    def calculate_best_view_angles(self, item):
        camera_position = self.get_camera_position()
        look_vector = np.array(
            [
                item["axisAlignedBoundingBox"]["center"]["x"] - camera_position["x"],
                item["axisAlignedBoundingBox"]["center"]["y"] - camera_position["y"],
                item["axisAlignedBoundingBox"]["center"]["z"] - camera_position["z"],
            ]
        )
        norm = np.linalg.norm(look_vector)
        look_vector = look_vector / norm if norm != 0 else look_vector
        yaw = np.arctan2(look_vector[0], look_vector[2])
        pitch = np.arcsin(look_vector[1])
        return np.degrees(yaw), np.degrees(pitch)
