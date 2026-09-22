"""AI2-THOR 模拟器封装：轨迹生成、动作执行、图像捕获

动作执行层委托给 _SimulatorAgent（继承自 RocAgent），直接复用
embodied_reasoner 中经过验证的 navigate + interact 逻辑：
  - compute_position_8：体积/面积感知的最优定位
  - BaseAction.pick_up：forceAction=True 保证拾取成功率
  - BaseAction.open/close：持物时先放地板，操作完再重拾
"""

import copy
import os
import math
import subprocess
from dataclasses import dataclass
from typing import Optional
import numpy as np
from PIL import Image

from ..schema import (
    TrajectoryStep,
    VisibleObject,
    AgentState,
    Position,
    ActionRecord,
    FeedbackRecord,
    MemoryCueExposure,
)
from .metadata import get_room_type
from .procthor import resolve_controller_scene
from .base_action import BaseAction
from .roc_agent import RocAgent


def _is_floor_id(object_id: str | None) -> bool:
    if not object_id:
        return False
    return object_id == "Floor" or object_id.startswith("Floor|")


_RENDERER_UNSAFE_OPEN_TARGETS = frozenset(
    {
        ("FloorPlan15", "Blinds|-01.41|+02.09|-00.30"),
    }
)


@dataclass(frozen=True)
class _GPUStat:
    index: int
    memory_used_mb: int
    memory_total_mb: int
    utilization: int

    @property
    def free_mb(self) -> int:
        return self.memory_total_mb - self.memory_used_mb


def _parse_gpu_index_list(raw: str | None) -> list[int]:
    if not raw:
        return []
    indices: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            indices.append(int(part))
        except ValueError:
            continue
    return indices


def _query_gpu_stats() -> list[_GPUStat]:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return []

    stats: list[_GPUStat] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            stats.append(
                _GPUStat(
                    index=int(parts[0]),
                    memory_used_mb=int(parts[1]),
                    memory_total_mb=int(parts[2]),
                    utilization=int(parts[3]),
                )
            )
        except ValueError:
            continue
    return stats


def _select_ai2thor_gpu() -> tuple[int | None, _GPUStat | None]:
    forced = os.environ.get("AI2THOR_RENDER_GPU_DEVICE")
    if forced is not None and forced.strip() != "":
        try:
            gpu_index = int(forced)
        except ValueError:
            print(
                f"[WARN] Invalid AI2THOR_RENDER_GPU_DEVICE={forced!r}; falling back to auto-selection"
            )
        else:
            match = None
            if os.environ.get("AI2THOR_RENDER_QUERY_FORCED_GPU_STATS") == "1":
                stats = _query_gpu_stats()
                match = next((gpu for gpu in stats if gpu.index == gpu_index), None)
            return gpu_index, match

    stats = _query_gpu_stats()
    if not stats:
        return None, None

    visible_gpu_indices = _parse_gpu_index_list(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if visible_gpu_indices:
        stats = [gpu for gpu in stats if gpu.index in visible_gpu_indices]
        if not stats:
            return None, None

    preferred_indices = _parse_gpu_index_list(
        os.environ.get("AI2THOR_RENDER_GPU_PREFER")
    )
    min_free_mb = int(os.environ.get("AI2THOR_RENDER_MIN_FREE_MB", "35000"))
    max_utilization = int(os.environ.get("AI2THOR_RENDER_MAX_UTILIZATION", "95"))

    def eligible(pool: list[_GPUStat]) -> list[_GPUStat]:
        return [
            gpu
            for gpu in pool
            if gpu.free_mb >= min_free_mb and gpu.utilization <= max_utilization
        ]

    preferred_pool = [
        gpu for idx in preferred_indices for gpu in stats if gpu.index == idx
    ]
    candidates = eligible(preferred_pool) or eligible(stats) or stats
    chosen = min(candidates, key=lambda gpu: (-gpu.free_mb, gpu.utilization, gpu.index))

    if visible_gpu_indices:
        return visible_gpu_indices.index(chosen.index), chosen
    return chosen.index, chosen


class _SimulatorAgent(RocAgent):
    """RocAgent 的轻量子类，专用于 TrajectoryGenerator。

    重写 navigate：跳过 adjust_view / adjust_height（蹲站切换对轨迹
    生成无意义），改由 TrajectoryGenerator 的 _look_at_object 精调朝向。
    重试上限 5 次，compute_position_8 返回 None 时退出循环。
    """

    def _safe_teleport(self, target_position, target_rotation):
        try:
            return self.action.action_mapping["teleport"](
                self.controller, position=target_position, rotation=target_rotation
            )
        except Exception as exc:
            print(f"  [WARN] teleport exception: {exc}")
            return None

    def navigate(self, item):
        target_position, target_rotation = self.compute_position_8(item, [])
        if target_position is None:
            return False, None, None

        event = self._safe_teleport(target_position, target_rotation)
        pre_target_positions = []
        max_retries = int(os.environ.get("SIM_NAV_MAX_RETRIES", "5"))
        while (
            event is None or not event.metadata["lastActionSuccess"]
        ) and max_retries > 0:
            print(f"  [WARN] teleport failed, retrying ({max_retries} left)...")
            pre_target_positions.append(target_position)
            target_position, target_rotation = self.compute_position_8(
                item, pre_target_positions
            )
            if target_position is None:
                break
            event = self._safe_teleport(target_position, target_rotation)
            self.update_event()
            max_retries -= 1

        return (
            bool(event and event.metadata["lastActionSuccess"]),
            target_position,
            target_rotation,
        )


class TrajectoryGenerator:
    """
    基于 AI2-THOR 的轨迹生成器。
    在模拟器中执行动作序列，捕获图像和状态。
    """

    def __init__(
        self,
        scene: str | dict,
        output_dir: str,
        width: int = 500,
        height: int = 500,
        fov: int = 90,
        headless: bool = True,
        allow_implicit_navigation: bool = True,
    ):
        self.scene = copy.deepcopy(scene)
        self._controller_scene = copy.deepcopy(resolve_controller_scene(scene))
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.allow_implicit_navigation = allow_implicit_navigation
        self._last_action_error = ""
        self.step_counter = 0
        os.makedirs(output_dir, exist_ok=True)

        from ai2thor.controller import Controller

        platform = None
        gpu_device = None
        if headless:
            try:
                from ai2thor.platform import CloudRendering

                platform = CloudRendering
            except ImportError:
                print("Warning: CloudRendering not available, using default platform")
            else:
                gpu_device, gpu_stat = _select_ai2thor_gpu()
                if gpu_device is not None:
                    if gpu_stat is not None:
                        print(
                            "[INFO] AI2-THOR CloudRendering selected "
                            f"gpu_device={gpu_device} "
                            f"(actual_gpu={gpu_stat.index}, free={gpu_stat.free_mb}MB, "
                            f"used={gpu_stat.memory_used_mb}MB, util={gpu_stat.utilization}%)"
                        )
                    else:
                        print(
                            f"[INFO] AI2-THOR CloudRendering selected gpu_device={gpu_device}"
                        )

        kwargs = dict(
            agentMode="default",
            visibilityDistance=1.5,
            scene=self._controller_scene,
            gridSize=0.25,
            snapToGrid=True,
            rotateStepDegrees=90,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            width=width,
            height=height,
            fieldOfView=fov,
        )
        if platform:
            kwargs["platform"] = platform
        if gpu_device is not None:
            kwargs["gpu_device"] = gpu_device
        self.controller = Controller(**kwargs)
        self._closed = False

        # Agent 在 Controller 启动后初始化，确保 last_event 已就绪
        self._agent = _SimulatorAgent(self.controller)
        self._reset_episode_state()

    def reset_episode(self, scene: str | dict, output_dir: str) -> None:
        """Load a new episode scene without replacing the Unity process."""
        if self._closed:
            raise RuntimeError("Cannot reset a closed TrajectoryGenerator")

        controller_scene = copy.deepcopy(resolve_controller_scene(scene))
        self.controller.reset(scene=controller_scene)

        self.scene = copy.deepcopy(scene)
        self._controller_scene = copy.deepcopy(controller_scene)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._agent = _SimulatorAgent(self.controller)
        self._reset_episode_state()

    def _reset_episode_state(self) -> None:
        self._last_action_error = ""
        self.step_counter = 0
        # Per-episode interaction hints must not leak across Controller.reset().
        self._last_pickup_info: dict = {}
        self._object_parent_hints: dict[str, str] = {}
        self._drawer_attachments: dict[str, dict[str, dict]] = {}
        self._last_navigation_target: Optional[str] = None

    def close(self):
        if not self._closed:
            self._closed = True
            self.controller.stop()

    def get_metadata(self) -> dict:
        return self.controller.last_event.metadata

    def get_visible_objects(self) -> list[dict]:
        return [obj for obj in self.get_metadata()["objects"] if obj["visible"]]

    def get_agent_state(self) -> AgentState:
        meta = self.get_metadata()
        agent = meta["agent"]
        held = None
        for obj in meta["objects"]:
            if obj.get("isPickedUp"):
                held = obj["objectId"]
                break
        return AgentState(
            position=Position(
                x=agent["position"]["x"],
                y=agent["position"]["y"],
                z=agent["position"]["z"],
            ),
            rotation_y=agent["rotation"]["y"],
            camera_horizon=agent["cameraHorizon"],
            holding=held,
        )

    def save_observation(self, prefix: str = "obs", caption: str = "") -> str:
        frame = self.controller.last_event.frame
        img = Image.fromarray(frame)
        if caption:
            img = self._add_caption(img, caption)
        filename = f"{prefix}_{self.step_counter:03d}.png"
        path = os.path.join(self.output_dir, filename)
        img.save(path)
        return filename

    @staticmethod
    def _add_caption(
        img: Image.Image, text: str, bar_height: int = 36, font_size: int = 18
    ) -> Image.Image:
        from PIL import ImageDraw, ImageFont

        w, h = img.size
        new_img = Image.new("RGB", (w, h + bar_height), (0, 0, 0))
        new_img.paste(img, (0, 0))
        draw = ImageDraw.Draw(new_img)
        font = None
        for font_path in [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except Exception:
                    continue
        if font is None:
            font = ImageFont.load_default()
        max_chars = w // (font_size // 2 + 1)
        display_text = text if len(text) <= max_chars else text[: max_chars - 2] + ".."
        bbox = draw.textbbox((0, 0), display_text, font=font)
        tw = bbox[2] - bbox[0]
        x = (w - tw) // 2
        y = h + (bar_height - font_size) // 2 - 2
        draw.text((x, y), display_text, fill=(255, 255, 255), font=font)
        return new_img

    def make_visible_objects_list(self) -> list[VisibleObject]:
        agent_pos = self.get_metadata()["agent"]["position"]
        result = []
        for obj in self.get_visible_objects():
            dist = math.sqrt(
                (obj["position"]["x"] - agent_pos["x"]) ** 2
                + (obj["position"]["z"] - agent_pos["z"]) ** 2
            )
            result.append(
                VisibleObject(
                    object_id=obj["objectId"],
                    object_type=obj["objectType"],
                    position=Position(
                        x=round(obj["position"]["x"], 3),
                        y=round(obj["position"]["y"], 3),
                        z=round(obj["position"]["z"], 3),
                    ),
                    distance=round(dist, 2),
                )
            )
        return result

    # ─── 导航与交互（委托给 _SimulatorAgent）─────────────────────

    def _look_at_object(self, object_id: str):
        """Teleport 到当前位置但调整 yaw/horizon 朝向目标物体。"""
        agent = self.get_metadata()["agent"]
        obj_meta = next(
            (o for o in self.get_metadata()["objects"] if o["objectId"] == object_id),
            None,
        )
        if not obj_meta:
            return
        ax, az = agent["position"]["x"], agent["position"]["z"]
        ox, oz = obj_meta["position"]["x"], obj_meta["position"]["z"]
        yaw = math.degrees(math.atan2(ox - ax, oz - az)) % 360
        eye_height = agent["position"]["y"] + 0.675
        obj_y = obj_meta["position"]["y"]
        dist_xz = math.sqrt((ox - ax) ** 2 + (oz - az) ** 2)
        horizon = (
            math.degrees(math.atan2(eye_height - obj_y, dist_xz))
            if dist_xz > 0.01
            else 30
        )
        horizon = max(-30, min(60, horizon))
        try:
            self.controller.step(
                action="Teleport",
                position=agent["position"],
                rotation=dict(x=0, y=yaw, z=0),
                horizon=horizon,
            )
        except Exception as exc:
            print(f"  [WARN] look-at teleport exception for {object_id}: {exc}")

    def _get_obj_meta(self, object_id: str) -> Optional[dict]:
        return next(
            (o for o in self.get_metadata()["objects"] if o["objectId"] == object_id),
            None,
        )

    def _set_action_error(self, message: str):
        self._last_action_error = message

    def _failure_feedback(self, default: str) -> str:
        return self._last_action_error or default

    @staticmethod
    def _object_id_position(object_id: str) -> Optional[dict]:
        parts = object_id.split("|")
        try:
            return {"x": float(parts[1]), "y": float(parts[2]), "z": float(parts[3])}
        except (IndexError, ValueError):
            return None

    def navigate_to_object(self, object_id: str) -> bool:
        """导航到物体附近并朝向物体。

        使用 _SimulatorAgent.navigate（即 compute_position_8）选出最优位置，
        再用 _look_at_object 精调相机朝向。
        如果 compute_position_8 找不到合法位置，退化到 AI2-THOR
        GetReachablePositions 的最近可达点。
        """
        obj_meta = self._get_obj_meta(object_id)
        if obj_meta is None:
            return False

        # 刷新 EventObject，确保体积/面积数据与当前场景一致
        self._agent.update_event()

        success, _, _ = self._agent.navigate(obj_meta)

        if not success:
            print(f"  [WARN] navigate failed for {object_id}, trying offset fallback")
            if self._navigate_to_nearest_reachable(object_id, obj_meta):
                self._last_navigation_target = object_id
                return True
            return False

        self._look_at_object(object_id)
        self._last_navigation_target = object_id
        return True

    def _navigate_to_nearest_reachable(self, object_id: str, obj_meta: dict) -> bool:
        pos = obj_meta["position"]
        try:
            event = self.controller.step(action="GetReachablePositions")
            reachable = event.metadata.get("actionReturn") or []
        except Exception as exc:
            print(f"  [WARN] GetReachablePositions exception for {object_id}: {exc}")
            reachable = []

        candidates = sorted(
            reachable,
            key=lambda p: (
                (float(p["x"]) - float(pos["x"])) ** 2
                + (float(p["z"]) - float(pos["z"])) ** 2
            ),
        )
        for candidate in candidates[:24]:
            try:
                self.controller.step(
                    action="Teleport",
                    position=candidate,
                    rotation=dict(x=0, y=0, z=0),
                    horizon=30,
                )
            except Exception as exc:
                print(f"  [WARN] reachable teleport exception for {object_id}: {exc}")
                continue
            if self.controller.last_event.metadata["lastActionSuccess"]:
                self._look_at_object(object_id)
                return True
        return False

    def _find_object_current_parent(self, object_id: str) -> Optional[str]:
        hinted_parent = self._object_parent_hints.get(object_id)
        objects = self.get_metadata()["objects"]
        target_obj = None
        for obj in objects:
            if obj["objectId"] == object_id:
                target_obj = obj
                break

        if target_obj and target_obj.get("isPickedUp"):
            return None

        parent_ids = (target_obj or {}).get("parentReceptacles") or []
        reverse_parent_ids = [
            obj["objectId"]
            for obj in objects
            if object_id in (obj.get("receptacleObjectIds") or [])
        ]

        # AI2-THOR can report multiple ancestors, e.g. [Floor, Safe] or
        # [CounterTop, Pot]. Prefer the receptacle we just placed into when it
        # is still backed by metadata, otherwise take the most specific parent.
        if hinted_parent and (
            hinted_parent in parent_ids or hinted_parent in reverse_parent_ids
        ):
            return hinted_parent

        for parent_id in reversed(parent_ids):
            if not _is_floor_id(parent_id):
                return parent_id

        for parent_id in reversed(reverse_parent_ids):
            if not _is_floor_id(parent_id):
                return parent_id

        return hinted_parent if not _is_floor_id(hinted_parent) else None

    def _force_pickup_for_setup(self, object_id: str) -> bool:
        self.controller.step(
            action="PickupObject",
            objectId=object_id,
            forceAction=True,
            manualInteract=False,
        )
        return self.controller.last_event.metadata["lastActionSuccess"]

    def _restore_validated_relocation_pose(
        self,
        object_id: str,
        dest_id: str,
        position: dict,
        rotation: dict,
    ) -> bool:
        if self.get_agent_state().holding == object_id:
            self.controller.step(action="DropHandObject", forceAction=True)

        self.controller.step(
            action="TeleportObject",
            objectId=object_id,
            position=position,
            rotation=rotation,
            forceAction=True,
            forceKinematic=True,
        )
        if self.controller.last_event.metadata["lastActionSuccess"]:
            self._object_parent_hints[object_id] = dest_id
            return True

        return False

    def _relocated_object_is_pickupable(self, object_id: str, dest_id: str) -> bool:
        if not self.navigate_to_object(object_id):
            return False

        obj_meta = self._get_obj_meta(object_id)
        if not obj_meta or not obj_meta.get("visible"):
            return False

        validated_position = dict(obj_meta["position"])
        validated_rotation = dict(obj_meta["rotation"])
        self.controller.step(
            action="PickupObject",
            objectId=object_id,
            forceAction=False,
            manualInteract=False,
        )
        if not self.controller.last_event.metadata["lastActionSuccess"]:
            return False

        return self._restore_validated_relocation_pose(
            object_id,
            dest_id,
            validated_position,
            validated_rotation,
        )

    def _forget_drawer_attachment(self, object_id: str):
        for contents in self._drawer_attachments.values():
            contents.pop(object_id, None)

    def _track_drawer_attachment(
        self,
        object_id: str,
        drawer_id: str,
        object_pos: dict,
        drawer_pos: dict,
        object_rot: dict,
    ):
        self._forget_drawer_attachment(object_id)
        self._drawer_attachments.setdefault(drawer_id, {})[object_id] = {
            "offset": {
                axis: object_pos[axis] - drawer_pos[axis] for axis in ("x", "y", "z")
            },
            "rotation": dict(object_rot),
        }
        self._object_parent_hints[object_id] = drawer_id

    def _sync_tracked_drawer_contents(self, drawer_id: str) -> bool:
        drawer_meta = self._get_obj_meta(drawer_id)
        if not drawer_meta:
            return False
        contents = self._drawer_attachments.get(drawer_id, {})
        success = True
        for object_id, info in list(contents.items()):
            obj_meta = self._get_obj_meta(object_id)
            if not obj_meta:
                contents.pop(object_id, None)
                continue
            if obj_meta.get("isPickedUp"):
                contents.pop(object_id, None)
                continue
            target_pos = {
                axis: drawer_meta["position"][axis] + info["offset"][axis]
                for axis in ("x", "y", "z")
            }
            event = self.controller.step(
                action="TeleportObject",
                objectId=object_id,
                position=target_pos,
                rotation=info.get("rotation") or obj_meta["rotation"],
                forceAction=True,
                forceKinematic=True,
            )
            success = success and event.metadata["lastActionSuccess"]
        return success

    def pickup_object(self, object_id: str) -> bool:
        """拾取物体。

        物体在容器内时：导航到容器 → 打开容器 → 直接 PickUp（不再导航到物体，
        避免容器内物体无可达位置导致 navigate 失败）。
        物体不在容器内时：正常导航到物体本身再 PickUp。
        """
        # 拾取前记录物体位置/旋转及所在抽屉的关闭坐标，供后续放回抽屉时使用
        # 注意：当容器已打开时 parentReceptacles 为空，需通过 receptacleObjectIds 反向查找
        obj_pre = self._get_obj_meta(object_id)
        if obj_pre:
            parent_id = self._find_object_current_parent(object_id)
            if not parent_id:
                # 容器开着时 parentReceptacles 为空，遍历 receptacleObjectIds 反查
                for obj in self.get_metadata()["objects"]:
                    if obj.get("openable") and object_id in (
                        obj.get("receptacleObjectIds") or []
                    ):
                        parent_id = obj["objectId"]
                        break
            parent_closed_pos = None
            parent_live_pos = None
            if parent_id:
                parent_closed_pos = self._object_id_position(parent_id)
                parent_obj = self._get_obj_meta(parent_id)
                if parent_obj:
                    parent_live_pos = dict(parent_obj["position"])
            self._last_pickup_info[object_id] = {
                "pos": dict(obj_pre["position"]),
                "rot": dict(obj_pre["rotation"]),
                "parent_drawer_id": parent_id,
                "parent_closed_pos": parent_closed_pos,
                "parent_live_pos": parent_live_pos,
            }

        current_parent = self._find_object_current_parent(object_id)
        if self.allow_implicit_navigation:
            if current_parent:
                parent_obj = self._get_obj_meta(current_parent)
                if (
                    parent_obj
                    and parent_obj.get("openable")
                    and not parent_obj.get("isOpen")
                ):
                    self.navigate_to_object(current_parent)
                    self.open_object(current_parent)
                # 已在容器旁，无需再导航到容器内的物体
            else:
                self.navigate_to_object(object_id)
        obj_meta = self._get_obj_meta(object_id)
        if obj_meta is None:
            self._set_action_error(
                f"Failed to pick up {object_id}: target object not found."
            )
            return False
        recently_navigated = object_id == self._last_navigation_target
        if not self.allow_implicit_navigation:
            parent_obj = self._get_obj_meta(current_parent) if current_parent else None
            if (
                parent_obj
                and parent_obj.get("openable")
                and not parent_obj.get("isOpen")
            ):
                self._set_action_error(
                    f"Failed to pick up {object_id}: it is inside closed {current_parent}. "
                    "Navigate to and open the receptacle first."
                )
                return False
            if not obj_meta.get("visible") and not recently_navigated:
                self._set_action_error(
                    f"Failed to pick up {object_id}: object is not visible or reachable. "
                    "Move closer, look around, or open its container first."
                )
                return False
            self.controller.step(
                action="PickupObject",
                objectId=object_id,
                forceAction=recently_navigated,
                manualInteract=False,
            )
        else:
            self._agent.interact(obj_meta, "pick_up")
        success = self.controller.last_event.metadata["lastActionSuccess"]
        if success:
            self._forget_drawer_attachment(object_id)
        else:
            err = (
                self.controller.last_event.metadata.get("errorMessage")
                or "object is not reachable"
            )
            self._set_action_error(
                f"Failed to pick up {object_id}: {err}. "
                "Move closer, look around, or open its container first."
            )
        return success

    def put_object(self, receptacle_id: str) -> bool:
        """放置手持物体到指定容器。

        若目标是抽屉（Drawer）且 PutObject 失败，自动 fallback 到
        teleport_object_into_drawer。放置动作不负责关闭容器；
        若任务需要关门，必须由动作序列显式执行 Close。
        """
        held_id = self.get_agent_state().holding
        if self.allow_implicit_navigation:
            self.navigate_to_object(receptacle_id)
        receptacle_meta = self._get_obj_meta(receptacle_id)
        if not held_id:
            self._set_action_error(
                "Failed to put object: the robot is not holding anything."
            )
            return False
        if receptacle_meta is None:
            self._set_action_error(
                f"Failed to put object: target receptacle {receptacle_id} not found."
            )
            return False
        recently_navigated = receptacle_id == self._last_navigation_target
        if (
            not self.allow_implicit_navigation
            and not receptacle_meta.get("visible")
            and not recently_navigated
        ):
            self._set_action_error(
                f"Failed to put object on {receptacle_id}: receptacle is not visible or reachable. "
                "Navigate to it first."
            )
            return False
        if self.allow_implicit_navigation:
            self._agent.interact(receptacle_meta, "put")
        else:
            self.controller.step(
                action="PutObject",
                objectId=receptacle_id,
                forceAction=True,
                placeStationary=True,
            )
        if self.controller.last_event.metadata["lastActionSuccess"]:
            if held_id:
                self._forget_drawer_attachment(held_id)
                self._object_parent_hints[held_id] = receptacle_id
            return True

        # PutObject 失败：若目标是抽屉则尝试 TeleportObject fallback.
        # ProcTHOR often represents drawer slots as parent ids such as
        # Dresser|...___3 with objectType=Drawer, so checking the id prefix alone
        # misses valid drawer targets.
        is_drawer_target = (
            receptacle_id.startswith("Drawer|")
            or receptacle_meta.get("objectType") == "Drawer"
        )
        if is_drawer_target:
            info = self._last_pickup_info.get(held_id) if held_id else None
            if info:
                print(
                    f"  [FALLBACK] PutObject→Drawer 失败，改用 TeleportObject: {held_id} → {receptacle_id}"
                )
                return self.teleport_object_into_drawer(
                    object_id=held_id,
                    drawer_id=receptacle_id,
                    ref_object_pos=info["pos"],
                    ref_drawer_closed_pos=info.get("parent_closed_pos"),
                    ref_drawer_live_pos=info.get("parent_live_pos"),
                    object_rot=info["rot"],
                )

        err = (
            self.controller.last_event.metadata.get("errorMessage")
            or "no valid placement was found"
        )
        if (
            held_id
            and "No valid positions" in err
            and self._can_surface_teleport_fallback(receptacle_meta)
        ):
            info = self._last_pickup_info.get(held_id) or {}
            print(
                "  [FALLBACK] PutObject→Surface 失败，改用 TeleportObject: "
                f"{held_id} → {receptacle_id}"
            )
            if self.teleport_object_onto_receptacle_surface(
                object_id=held_id,
                receptacle_id=receptacle_id,
                object_rot=info.get("rot"),
            ):
                return True

        self._set_action_error(f"Failed to put object on {receptacle_id}: {err}.")
        return False

    @staticmethod
    def _can_surface_teleport_fallback(receptacle_meta: dict) -> bool:
        if not receptacle_meta or not receptacle_meta.get("receptacle"):
            return False
        object_type = str(receptacle_meta.get("objectType") or "")
        return object_type in {
            "ArmChair",
            "BathtubBasin",
            "Bed",
            "Cabinet",
            "CoffeeTable",
            "CounterTop",
            "Desk",
            "DiningTable",
            "Dresser",
            "Ottoman",
            "Shelf",
            "ShelvingUnit",
            "SideTable",
            "Sofa",
            "Stool",
            "TVStand",
        }

    def teleport_object_onto_receptacle_surface(
        self,
        object_id: str,
        receptacle_id: str,
        object_rot: Optional[dict] = None,
        validate_pickup: bool = False,
    ) -> bool:
        """Place an object on a chosen surface when AI2-THOR has no sampled pose.

        This is only used after the model selected a visible, receptacle-like
        surface and AI2-THOR returned "No valid positions". It repairs simulator
        pose sampling, not task semantics: the selected receptacle remains the
        parent hint used by the evaluator.
        """
        receptacle_meta = self._get_obj_meta(receptacle_id)
        object_meta = self._get_obj_meta(object_id)
        if not receptacle_meta or not object_meta:
            return False

        if self.get_agent_state().holding:
            self.controller.step(action="DropHandObject", forceAction=True)
        if self.get_agent_state().holding:
            floor_id = next(
                (
                    obj["objectId"]
                    for obj in self.get_metadata()["objects"]
                    if obj.get("objectType") == "Floor"
                ),
                None,
            )
            if floor_id:
                self.controller.step(
                    action="PutObject",
                    objectId=floor_id,
                    forceAction=True,
                    placeStationary=True,
                )
        if self.get_agent_state().holding:
            return False

        receptacle_box = receptacle_meta.get("axisAlignedBoundingBox") or {}
        object_box = object_meta.get("axisAlignedBoundingBox") or {}
        rec_center = dict(receptacle_box.get("center") or receptacle_meta["position"])
        rec_size = dict(receptacle_box.get("size") or {})
        obj_size = dict(object_box.get("size") or {})
        top_y = (
            float(rec_center.get("y", receptacle_meta["position"]["y"]))
            + float(rec_size.get("y", 0.12)) / 2.0
            + float(obj_size.get("y", 0.10)) / 2.0
            + 0.015
        )
        max_dx = min(max(float(rec_size.get("x", 0.20)) * 0.25, 0.03), 0.16)
        max_dz = min(max(float(rec_size.get("z", 0.20)) * 0.25, 0.03), 0.16)
        target_positions = []
        for dx, dz in (
            (0.0, 0.0),
            (max_dx, 0.0),
            (-max_dx, 0.0),
            (0.0, max_dz),
            (0.0, -max_dz),
            (max_dx, max_dz),
            (-max_dx, max_dz),
            (max_dx, -max_dz),
            (-max_dx, -max_dz),
        ):
            target_positions.append(
                {
                    "x": float(rec_center.get("x", receptacle_meta["position"]["x"]))
                    + dx,
                    "y": top_y,
                    "z": float(rec_center.get("z", receptacle_meta["position"]["z"]))
                    + dz,
                }
            )

        rotation = object_rot or object_meta.get("rotation") or {"x": 0, "y": 0, "z": 0}
        for target_pos in target_positions:
            event = self.controller.step(
                action="TeleportObject",
                objectId=object_id,
                position=target_pos,
                rotation=rotation,
                forceAction=True,
                forceKinematic=True,
            )
            if event.metadata["lastActionSuccess"]:
                self._forget_drawer_attachment(object_id)
                self._object_parent_hints[object_id] = receptacle_id
                if validate_pickup and not self._relocated_object_is_pickupable(
                    object_id,
                    receptacle_id,
                ):
                    continue
                return True
        return False

    def open_object(self, object_id: str) -> bool:
        """打开容器。已经打开时视为成功，不再扰动手中物体。"""
        obj = self._get_obj_meta(object_id)
        if not obj or not obj.get("openable"):
            return False
        if obj.get("isOpen"):
            return True
        if (
            isinstance(self.scene, str)
            and (self.scene.removesuffix("_physics"), object_id)
            in _RENDERER_UNSAFE_OPEN_TARGETS
        ):
            self._set_action_error(
                f"Failed to open {object_id}: renderer-unsafe AI2-THOR interaction."
            )
            return False
        held_id = self.get_agent_state().holding
        if held_id:
            floor_id = next(
                (
                    item["objectId"]
                    for item in self.get_metadata()["objects"]
                    if item.get("objectType") == "Floor"
                ),
                None,
            )
            if floor_id:
                self.controller.step(
                    action="PutObject",
                    objectId=floor_id,
                    forceAction=True,
                    placeStationary=True,
                )
            if self.get_agent_state().holding:
                self.controller.step(action="DropHandObject", forceAction=True)
        self.controller.step(
            action="OpenObject",
            objectId=object_id,
            openness=1,
            forceAction=(object_id == self._last_navigation_target),
        )
        success = self.controller.last_event.metadata["lastActionSuccess"]
        if success:
            success = self._sync_tracked_drawer_contents(object_id)
        if held_id:
            self.controller.step(
                action="PickupObject",
                objectId=held_id,
                forceAction=True,
                manualInteract=False,
            )
            success = (
                success and self.controller.last_event.metadata["lastActionSuccess"]
            )
        return success

    def close_object(self, object_id: str) -> bool:
        """关闭容器。已经关闭时视为成功，不再重复执行底层 CloseObject。"""
        obj = self._get_obj_meta(object_id)
        if not obj or not obj.get("openable"):
            return False
        if not obj.get("isOpen"):
            return True
        BaseAction.close(
            self.controller,
            object_id,
            force_action=(object_id == self._last_navigation_target),
        )
        success = self.controller.last_event.metadata["lastActionSuccess"]
        if success:
            success = self._sync_tracked_drawer_contents(object_id)
        return success

    def teleport_object_into_drawer(
        self,
        object_id: str,
        drawer_id: str,
        ref_object_pos: dict,
        ref_drawer_closed_pos: Optional[dict],
        ref_drawer_live_pos: Optional[dict],
        object_rot: dict,
    ) -> bool:
        """将物体传送进抽屉（绕开 PutObject 的采样限制）。

        PutObject 对细长物体（Knife）在抽屉里会报 "No valid positions"，
        本方法改用 TeleportObject 完成放置，并保持抽屉打开：
          1. 若手中持有物体则先放地板
          2. 要求目标抽屉已由显式 Open 步骤打开
          3. 将物体相对源抽屉实时坐标的 x/y/z 偏移，映射到目标抽屉
          4. TeleportObject 到目标位置
        Close 必须作为后续显式动作记录，避免 PutObject 步隐式关门。
        """
        drawer_meta = self._get_obj_meta(drawer_id)
        if not drawer_meta or not drawer_meta.get("isOpen"):
            return False
        drawer_pos = dict(drawer_meta["position"])
        ref_drawer_pos = ref_drawer_live_pos or ref_drawer_closed_pos
        if not ref_drawer_pos:
            return False

        state = self.get_agent_state()
        if state.holding:
            floor_id = next(
                (
                    o["objectId"]
                    for o in self.get_metadata()["objects"]
                    if o["objectType"] == "Floor"
                ),
                None,
            )
            if floor_id:
                self.controller.step(
                    action="PutObject",
                    objectId=floor_id,
                    forceAction=True,
                    placeStationary=True,
                )
            if self.get_agent_state().holding:
                self.controller.step(action="DropHandObject", forceAction=True)
        if ref_drawer_pos:
            target_positions = [
                {
                    axis: drawer_pos[axis]
                    + (ref_object_pos[axis] - ref_drawer_pos[axis])
                    for axis in ("x", "y", "z")
                }
            ]
        else:
            target_positions = []
        # If the object came from a surface rather than another drawer, there is
        # no useful source-drawer frame. Try compact positions near the drawer
        # center; force TeleportObject is enough for benchmark state grounding.
        for dx, dy, dz in (
            (0.0, 0.0, 0.0),
            (0.0, 0.03, 0.0),
            (0.03, 0.0, 0.0),
            (-0.03, 0.0, 0.0),
            (0.0, 0.0, 0.03),
            (0.0, 0.0, -0.03),
        ):
            target_positions.append(
                {
                    "x": drawer_pos["x"] + dx,
                    "y": drawer_pos["y"] + dy,
                    "z": drawer_pos["z"] + dz,
                }
            )

        target_pos = None
        for candidate_pos in target_positions:
            e = self.controller.step(
                action="TeleportObject",
                objectId=object_id,
                position=candidate_pos,
                rotation=object_rot,
                forceAction=True,
                forceKinematic=True,
            )
            if e.metadata["lastActionSuccess"]:
                target_pos = candidate_pos
                break
        if target_pos is None:
            return False

        self._track_drawer_attachment(
            object_id=object_id,
            drawer_id=drawer_id,
            object_pos=target_pos,
            drawer_pos=drawer_pos,
            object_rot=object_rot,
        )
        return True

    def _discard_steps(self, steps: list, counter_before: int):
        """删除失败任务批次产生的图片，并回滚 step_counter 使编号连续。"""
        for step in steps:
            img = os.path.join(self.output_dir, step.image_path)
            if os.path.exists(img):
                os.remove(img)
        self.step_counter = counter_before

    def ensure_hand_empty(self):
        """确保 agent 手中没有物体。"""
        state = self.get_agent_state()
        if state.holding is None:
            return
        held_id = state.holding
        held_type = held_id.split("|")[0]
        print(f"    [CLEANUP] 手中有 {held_type}，尝试放下")

        for obj in self.get_visible_objects():
            if obj.get("receptacle") and obj["objectType"] != "Floor":
                self.controller.step(
                    action="PutObject",
                    objectId=obj["objectId"],
                    forceAction=True,
                    placeStationary=True,
                )
                if self.controller.last_event.metadata["lastActionSuccess"]:
                    print(f"    [CLEANUP] 放到了 {obj['objectType']} 上")
                    return

        for fallback_type in [
            "CounterTop",
            "DiningTable",
            "CoffeeTable",
            "SideTable",
            "Desk",
            "Shelf",
            "Bed",
            "Sofa",
        ]:
            for obj in self.get_metadata()["objects"]:
                if obj["objectType"] == fallback_type and obj.get("receptacle"):
                    self.navigate_to_object(obj["objectId"])
                    self.controller.step(
                        action="PutObject",
                        objectId=obj["objectId"],
                        forceAction=True,
                        placeStationary=True,
                    )
                    if self.controller.last_event.metadata["lastActionSuccess"]:
                        print(f"    [CLEANUP] 导航并放到了 {obj['objectType']} 上")
                        return

        self.controller.step(action="DropHandObject", forceAction=True)
        print(f"    [CLEANUP] 强制丢弃")

    def relocate_object_out_of_receptacle_type(
        self,
        object_id: str,
        receptacle_type: str,
    ) -> Optional[str]:
        """Move an eval target out of an already-correct receptacle type.

        This is a reset-time setup helper, not a model action. It prevents a
        probe from starting in a trivially solved state, e.g. the task knife is
        already inside a Drawer before the model acts.
        """
        obj = self._get_obj_meta(object_id)
        if not obj:
            return None

        parent_id = self._find_object_current_parent(object_id)
        parent = self._get_obj_meta(parent_id) if parent_id else None
        parent_was_open = bool(parent and parent.get("isOpen"))

        if parent and parent.get("openable") and not parent.get("isOpen"):
            self.controller.step(
                action="OpenObject",
                objectId=parent_id,
                openness=1,
                forceAction=True,
            )

        self.controller.step(
            action="PickupObject",
            objectId=object_id,
            forceAction=True,
            manualInteract=False,
        )
        if not self.controller.last_event.metadata["lastActionSuccess"]:
            if parent and parent.get("openable") and not parent_was_open:
                self.controller.step(
                    action="CloseObject",
                    objectId=parent_id,
                    forceAction=True,
                )
            return None

        for dest in self._relocation_receptacle_candidates(receptacle_type):
            self.navigate_to_object(dest["objectId"])
            self.controller.step(
                action="PutObject",
                objectId=dest["objectId"],
                forceAction=True,
                placeStationary=True,
            )
            if self.controller.last_event.metadata["lastActionSuccess"]:
                self._forget_drawer_attachment(object_id)
                self._object_parent_hints[object_id] = dest["objectId"]
                if self._relocated_object_is_pickupable(object_id, dest["objectId"]):
                    if parent and parent.get("openable") and not parent_was_open:
                        self.controller.step(
                            action="CloseObject",
                            objectId=parent_id,
                            forceAction=True,
                        )
                    return dest["objectId"]
                if not self._force_pickup_for_setup(object_id):
                    break

        self.controller.step(action="DropHandObject", forceAction=True)
        if parent and parent.get("openable") and not parent_was_open:
            self.controller.step(
                action="CloseObject",
                objectId=parent_id,
                forceAction=True,
            )
        return None

    def _relocation_receptacle_candidates(self, excluded_type: str) -> list[dict]:
        priority = {
            "CounterTop": 0,
            "DiningTable": 1,
            "CoffeeTable": 2,
            "SideTable": 3,
            "Desk": 4,
            "Shelf": 5,
            "StoveBurner": 6,
            "SinkBasin": 7,
        }
        candidates = []
        for obj in self.get_metadata()["objects"]:
            obj_type = obj.get("objectType")
            if not obj.get("receptacle"):
                continue
            if obj_type in {excluded_type, "Floor", "GarbageCan"}:
                continue
            if obj.get("openable"):
                continue
            candidates.append(obj)
        return sorted(
            candidates,
            key=lambda o: (
                priority.get(o.get("objectType"), 100),
                not bool(o.get("visible")),
                o.get("objectId", ""),
            ),
        )

    def toggle_on(self, object_id: str) -> bool:
        self.controller.step(action="ToggleObjectOn", objectId=object_id)
        return self.controller.last_event.metadata["lastActionSuccess"]

    def toggle_off(self, object_id: str) -> bool:
        self.controller.step(action="ToggleObjectOff", objectId=object_id)
        return self.controller.last_event.metadata["lastActionSuccess"]

    def transfer_contents(
        self, _source_id: Optional[str], target_id: Optional[str]
    ) -> bool:
        if not target_id:
            self._set_action_error(
                "Failed to transfer contents: no target receptacle was provided."
            )
            return False
        if self.allow_implicit_navigation:
            self.navigate_to_object(target_id)
            return True
        target_meta = self._get_obj_meta(target_id)
        if not target_meta:
            self._set_action_error(
                f"Failed to transfer contents: target {target_id} not found."
            )
            return False
        if not target_meta.get("visible"):
            self._set_action_error(
                f"Failed to transfer contents to {target_id}: target is not visible or reachable. "
                "Navigate to it first."
            )
            return False
        return True

    def rotate_agent(self, degrees: float) -> bool:
        if degrees > 0:
            self.controller.step(action="RotateRight", degrees=abs(degrees))
        else:
            self.controller.step(action="RotateLeft", degrees=abs(degrees))
        return self.controller.last_event.metadata["lastActionSuccess"]

    def move_forward(self) -> bool:
        self.controller.step(action="MoveAhead")
        return self.controller.last_event.metadata["lastActionSuccess"]

    # ─── 复合动作（生成轨迹步骤）──────────────────────────────────

    def execute_and_record(
        self,
        action_type: str,
        target: Optional[str] = None,
        instrument: Optional[str] = None,
        nl_description: str = "",
        simulate_failure: bool = False,
        failure_message: str = "",
        event_memory_tag: Optional[str] = None,
        memory_cue: Optional[MemoryCueExposure] = None,
    ) -> TrajectoryStep:
        success = True
        feedback_msg = ""
        self._last_action_error = ""

        if not simulate_failure:
            if action_type == "Navigate":
                success = self.navigate_to_object(target)
                feedback_msg = (
                    f"Navigated to {target}"
                    if success
                    else f"Failed to navigate to {target}"
                )
            elif action_type == "PickUp":
                success = self.pickup_object(target)
                feedback_msg = (
                    f"Picked up {target}"
                    if success
                    else self._failure_feedback(f"Failed to pick up {target}")
                )
            elif action_type == "PutObject":
                success = self.put_object(target)
                feedback_msg = (
                    f"Put object on {target}"
                    if success
                    else self._failure_feedback("Failed to put object")
                )
            elif action_type == "Open":
                success = self.open_object(target)
                if not success and self.allow_implicit_navigation:
                    self.navigate_to_object(target)
                    success = self.open_object(target)
                feedback_msg = (
                    f"Opened {target}"
                    if success
                    else self._failure_feedback(
                        f"Failed to open {target}. Navigate to it first or check whether it is already open."
                    )
                )
            elif action_type == "Close":
                success = self.close_object(target)
                if not success and self.allow_implicit_navigation:
                    self.navigate_to_object(target)
                    success = self.close_object(target)
                feedback_msg = (
                    f"Closed {target}"
                    if success
                    else self._failure_feedback(
                        f"Failed to close {target}. Navigate to it first or check whether it is already closed."
                    )
                )
            elif action_type == "ToggleOn":
                success = self.toggle_on(target)
                if not success and self.allow_implicit_navigation:
                    self.navigate_to_object(target)
                    success = self.toggle_on(target)
                feedback_msg = (
                    f"Toggled on {target}"
                    if success
                    else self._failure_feedback(
                        f"Failed to toggle on {target}. Navigate to it first."
                    )
                )
            elif action_type == "ToggleOff":
                success = self.toggle_off(target)
                if not success and self.allow_implicit_navigation:
                    self.navigate_to_object(target)
                    success = self.toggle_off(target)
                feedback_msg = (
                    f"Toggled off {target}"
                    if success
                    else self._failure_feedback(
                        f"Failed to toggle off {target}. Navigate to it first."
                    )
                )
            elif action_type == "TransferContents":
                success = self.transfer_contents(instrument, target)
                feedback_msg = (
                    f"Transferred contents from {instrument or 'held object'} to {target}"
                    if success
                    else self._failure_feedback("Failed to transfer contents")
                )
            elif action_type == "RotateRight":
                success = self.rotate_agent(90)
                feedback_msg = "Rotated right"
            elif action_type == "RotateLeft":
                success = self.rotate_agent(-90)
                feedback_msg = "Rotated left"
            elif action_type == "MoveForward":
                success = self.move_forward()
                feedback_msg = "Moved forward" if success else "Blocked"
            elif action_type == "HumanIntervention":
                success = True
                feedback_msg = nl_description or "Human intervention"
            else:
                feedback_msg = f"Unknown action: {action_type}"
                success = False
        else:
            if target and self.allow_implicit_navigation:
                self.navigate_to_object(target)
            success = False
            feedback_msg = failure_message

        status = "OK" if success else "FAIL"
        caption = f"[Step {self.step_counter}] {nl_description}  [{status}]"
        img_path = self.save_observation(caption=caption)

        step = TrajectoryStep(
            step_id=self.step_counter,
            image_path=img_path,
            visible_objects=self.make_visible_objects_list(),
            agent_state=self.get_agent_state(),
            action=ActionRecord(
                action_type=action_type,
                target=target,
                instrument=instrument,
                natural_language=nl_description,
            ),
            feedback=FeedbackRecord(
                success=success,
                message=feedback_msg,
                event_memory_tag=event_memory_tag,
            ),
            memory_cue_exposed=memory_cue,
        )
        self.step_counter += 1
        return step

    def put_object_with_fallback(
        self,
        receptacle_ids: list[str],
        obj_nl: str = "",
    ) -> list[TrajectoryStep]:
        """依次尝试将手持物体放到候选容器中，成功时只保留该次的步骤。

        每次尝试：Navigate → PutObject；失败则删除图片、回滚 step_counter，继续下一候选。
        全部失败返回空列表（物体仍在手中，由调用方的后续步骤处理）。
        """
        for receptacle_id in receptacle_ids:
            receptacle_type = receptacle_id.split("|")[0]
            counter_before = self.step_counter
            nav_step = self.execute_and_record(
                action_type="Navigate",
                target=receptacle_id,
                nl_description=f"Navigate to {receptacle_type} to return {obj_nl}",
            )
            put_step = self.execute_and_record(
                action_type="PutObject",
                target=receptacle_id,
                nl_description=f"Place {obj_nl} on {receptacle_type} (restored)",
            )
            if put_step.feedback.success:
                return [nav_step, put_step]
            self._discard_steps([nav_step, put_step], counter_before)
        return []

    def generate_noise_steps(self, n_steps: int = 10) -> list[TrajectoryStep]:
        steps = []
        actions = ["MoveForward", "RotateRight", "RotateLeft"]
        for _ in range(n_steps):
            action = actions[np.random.randint(len(actions))]
            step = self.execute_and_record(
                action_type=action,
                nl_description=f"(noise action) {action}",
            )
            steps.append(step)
        return steps

    def generate_l1_noise(
        self, n_tasks: int = 3, seed: Optional[int] = None
    ) -> tuple[list[TrajectoryStep], list[str]]:
        from ..tasks import sample_tasks

        all_steps: list[TrajectoryStep] = []
        ok_descriptions: list[str] = []
        rng = np.random.RandomState(seed)
        used_objects: set[str] = set()

        l1_retries = n_tasks * 4
        for _ in range(l1_retries):
            if len(ok_descriptions) >= n_tasks:
                break

            live_meta = self.get_metadata()
            tasks = sample_tasks(
                live_meta,
                level="L1",
                n_tasks=5,
                seed=int(rng.randint(0, 100000)),
            )
            task = None
            for t in tasks:
                if t.key_object not in used_objects:
                    task = t
                    break
            if task is None:
                break

            action_dicts = [s.to_dict() for s in task.steps]
            counter_before = self.step_counter
            sub_steps = self.generate_task_trajectory(action_dicts)
            has_failure = any(not s.feedback.success for s in sub_steps)

            if has_failure:
                print(f"    [SKIP] L1任务失败: {task.name}")
                self._discard_steps(sub_steps, counter_before)
                self.ensure_hand_empty()
                used_objects.add(task.key_object)
                continue

            all_steps.extend(sub_steps)
            ok_descriptions.append(f"[{task.task_type}] {task.name}")
            used_objects.add(task.key_object)

        remaining = n_tasks - len(ok_descriptions)
        if remaining > 0:
            print(
                f"    [INFO] L1 完成 {len(ok_descriptions)} 个，用 L0 补 {remaining} 个"
            )
            l0_used: set[str] = set()
            l0_retries = remaining * 5
            for _ in range(l0_retries):
                if len(ok_descriptions) >= n_tasks:
                    break

                live_meta = self.get_metadata()
                tasks = sample_tasks(
                    live_meta,
                    level="L0",
                    n_tasks=10,
                    seed=int(rng.randint(0, 100000)),
                )
                task = None
                for t in tasks:
                    if t.key_object not in l0_used:
                        task = t
                        break
                if task is None:
                    break

                action_dicts = [s.to_dict() for s in task.steps]
                counter_before = self.step_counter
                sub_steps = self.generate_task_trajectory(action_dicts)
                if any(not s.feedback.success for s in sub_steps):
                    self._discard_steps(sub_steps, counter_before)
                    self.ensure_hand_empty()
                    l0_used.add(task.key_object)
                    continue

                all_steps.extend(sub_steps)
                ok_descriptions.append(f"[L0] {task.name}")
                l0_used.add(task.key_object)

        if not all_steps:
            return self.generate_noise_steps(15), ["random navigation"]

        return all_steps, ok_descriptions

    @staticmethod
    def _is_close_for(action_dict: Optional[dict], object_id: str) -> bool:
        return bool(
            action_dict
            and action_dict.get("action") == "Close"
            and action_dict.get("target") == object_id
        )

    def _expand_action_for_container(
        self,
        action_dict: dict,
        next_action_dict: Optional[dict] = None,
    ) -> list[dict]:
        """Navigate / PickUp 的目标在关闭容器内时，自动在前面插入 Navigate→Open。

        - Navigate→Object：展开为 Navigate→Container + Open→Container（丢弃原 Navigate→Object，
          容器内物体通常无独立可达位置）。
        - PickUp→Object：展开为 Navigate→Container + Open→Container + PickUp→Object + Close→Container
          关闭操作由此处统一插入，不在 pickup_object 内隐式执行，确保 Close 显式出现在轨迹中。
          若容器已经是开着的（例如模板层已显式执行了 Open 步骤），则不插入 Open/Close，
          避免与模板中已有的 Close 步骤重复。
        """
        atype = action_dict.get("action", "")
        target = action_dict.get("target")
        if atype not in ("Navigate", "PickUp") or not target:
            return [action_dict]
        parent_id = self._find_object_current_parent(target)
        if not parent_id:
            return [action_dict]
        parent_obj = self._get_obj_meta(parent_id)
        if not parent_obj or not parent_obj.get("openable"):
            return [action_dict]
        parent_type = parent_id.split("|")[0]
        target_type = target.split("|")[0]
        self._object_parent_hints[target] = parent_id

        if parent_obj.get("isOpen"):
            if atype == "PickUp" and not self._is_close_for(
                next_action_dict, parent_id
            ):
                return [
                    action_dict,
                    {
                        "action": "Close",
                        "target": parent_id,
                        "nl": f"Close {parent_type}",
                    },
                ]
            return [action_dict]

        prefix = [
            {
                "action": "Navigate",
                "target": parent_id,
                "nl": f"Navigate to {parent_type} ({target_type} is inside)",
            },
            {"action": "Open", "target": parent_id, "nl": f"Open {parent_type}"},
        ]
        if atype == "Navigate":
            return prefix  # 不需要再 Navigate 到容器内物体
        # PickUp：插入 Open 前缀，并在 PickUp 后追加 Close，使关门出现在轨迹中
        suffix = []
        if not self._is_close_for(next_action_dict, parent_id):
            suffix.append(
                {"action": "Close", "target": parent_id, "nl": f"Close {parent_type}"}
            )
        return prefix + [action_dict] + suffix

    def generate_task_trajectory(
        self, task_actions: list[dict]
    ) -> list[TrajectoryStep]:
        steps = []
        for idx, ta in enumerate(task_actions):
            next_ta = task_actions[idx + 1] if idx + 1 < len(task_actions) else None
            for action in self._expand_action_for_container(ta, next_ta):
                step = self.execute_and_record(
                    action_type=action["action"],
                    target=action.get("target"),
                    instrument=action.get("instrument"),
                    nl_description=action.get("nl", ""),
                    simulate_failure=action.get("simulate_failure", False),
                    failure_message=action.get("failure_message", ""),
                    event_memory_tag=action.get("event_memory_tag"),
                    memory_cue=action.get("memory_cue"),
                )
                steps.append(step)
                if not step.feedback.success and not action.get("simulate_failure"):
                    return steps
        return steps

    def generate_leave_room_step(self, dest_room: str) -> TrajectoryStep:
        best_rotation = None
        min_visible = 999
        for _ in range(4):
            self.controller.step(action="RotateRight")
            vis_count = len(self.get_visible_objects())
            if vis_count < min_visible:
                min_visible = vis_count
                best_rotation = self.get_metadata()["agent"]["rotation"]["y"]
        if best_rotation is not None:
            agent = self.get_metadata()["agent"]
            try:
                self.controller.step(
                    action="Teleport",
                    position=agent["position"],
                    rotation=dict(x=0, y=best_rotation, z=0),
                    horizon=0,
                )
            except Exception as exc:
                print(f"  [WARN] leave-room teleport exception: {exc}")

        room_name = get_room_type(self.scene)
        caption = f"[Step {self.step_counter}] LeaveRoom: leave {room_name} and go to {dest_room}"
        img_path = self.save_observation(prefix="transition", caption=caption)
        step = TrajectoryStep(
            step_id=self.step_counter,
            image_path=img_path,
            visible_objects=self.make_visible_objects_list(),
            agent_state=self.get_agent_state(),
            action=ActionRecord(
                action_type="LeaveRoom",
                target=dest_room,
                natural_language=f"Leave {room_name} and go to {dest_room}",
            ),
            feedback=FeedbackRecord(
                success=True,
                message=f"Leaving {room_name}, heading to {dest_room}.",
            ),
        )
        self.step_counter += 1
        return step

    def generate_enter_room_step(self, from_room: str) -> TrajectoryStep:
        room_name = get_room_type(self.scene)
        caption = f"[Step {self.step_counter}] EnterRoom: arrive in {room_name} from {from_room}"
        img_path = self.save_observation(prefix="transition", caption=caption)
        step = TrajectoryStep(
            step_id=self.step_counter,
            image_path=img_path,
            visible_objects=self.make_visible_objects_list(),
            agent_state=self.get_agent_state(),
            action=ActionRecord(
                action_type="EnterRoom",
                target=room_name,
                natural_language=f"Arrive in {room_name} from {from_room}",
            ),
            feedback=FeedbackRecord(
                success=True,
                message=f"Arrived at {room_name} from {from_room}.",
            ),
        )
        self.step_counter += 1
        return step
