"""Environment wrapper for EmbodiedMemorizerBench probes.

The goal is to expose a small benchmark-native interface for external planners:

    obs = env.reset()
    img_path = obs["image_path"]
    action = planner.act(img_path, obs["instruction"])
    obs, reward, done, info = env.step(action)

Actions can be either:
  - an integer action_id from obs["available_actions"]
  - a parameterized dict: {"action_type": "Navigate", "target": "..."}

This file intentionally stays thin: TrajectoryGenerator executes actions,
EpisodeEvaluator scores the accumulated action trace.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .evaluator import EpisodeEvaluator
from .matchers import object_type, split_type_list, targets_match

if TYPE_CHECKING:
    from emem_bench.construction.scene_utils.simulator import TrajectoryGenerator


DEFAULT_ACTION_TYPES = (
    "Navigate",
    "PickUp",
    "PutObject",
    "Open",
    "Close",
    "ToggleOn",
    "ToggleOff",
    "TransferContents",
    "RotateLeft",
    "RotateRight",
    "MoveForward",
    "MemoryInsufficient",
    "Done",
    "Stop",
)

STRUCTURAL_ACTION_EXCLUDE_TYPES = {
    "Ceiling",
    "Doorframe",
    "Doorway",
    "Floor",
    "Wall",
    "Window",
}


class RendererUnavailableError(RuntimeError):
    """The evaluator's renderer failed; the rollout decides whether to recover."""


class EmbodiedMemorizerEnv:
    """Probe-level environment wrapper for generated benchmark episodes."""

    def __init__(
        self,
        episode_path: str | None = None,
        episode_dir: str | None = None,
        *,
        episode_paths: list[str] | None = None,
        log_path: str = "running/embodied_memorizer",
        start_epi_index: int = 0,
        max_steps: int | None = None,
        width: int = 500,
        height: int = 500,
        headless: bool = True,
        include_all_targets: bool = True,
        strict_actions: bool = True,
        expose_pickupable_navigation: bool = False,
        strict_l2_instance_matching: bool = False,
        expose_l2_task_object_label: bool = False,
    ):
        if episode_paths is not None and (episode_path or episode_dir):
            raise ValueError(
                "episode_paths cannot be combined with episode_path or episode_dir"
            )
        resolved_episode_paths = self._collect_episode_paths(
            episode_path,
            episode_dir,
            episode_paths,
        )
        if not resolved_episode_paths:
            raise ValueError(
                "No episode json found. Provide episode_path or episode_dir."
            )

        self.log_path = log_path
        self.width = width
        self.height = height
        self.headless = headless
        self.include_all_targets = include_all_targets
        self.strict_actions = strict_actions
        self.expose_pickupable_navigation = expose_pickupable_navigation
        self.strict_l2_instance_matching = strict_l2_instance_matching
        self.expose_l2_task_object_label = expose_l2_task_object_label
        self._max_steps = max_steps

        self.eval_items = self._build_eval_items(resolved_episode_paths)
        self.number_of_episodes = len(self.eval_items)
        self._current_episode_num = start_epi_index
        self._current_step = 0
        self._cur_invalid_actions = 0

        self.evaluator: Optional[EpisodeEvaluator] = None
        self.probe: Optional[dict] = None
        self.sim: Optional[TrajectoryGenerator] = None
        self.episode_log: list[dict] = []
        self.model_actions: list[dict] = []
        self.available_actions: list[dict] = []
        self.object_labels: dict[str, str] = {}
        self._last_result = None
        self._last_observation: dict | None = None
        self._episode_output_dir = ""
        self._task_object_id: Optional[str] = None
        self._last_navigation_target: Optional[str] = None
        self._initial_state_adjustments: list[dict] = []
        self._renderer_circuit_open = False
        self._renderer_failure: str | None = None

    # ─── Episode lifecycle ─────────────────────────────────

    def reset(self) -> dict:
        """Start the current probe episode and return the initial observation."""
        if self._current_episode_num >= self.number_of_episodes:
            raise StopIteration("No more evaluation episodes.")
        if self._renderer_circuit_open:
            raise RendererUnavailableError(
                f"Renderer circuit is open: {self._renderer_failure or 'unknown failure'}"
            )

        item = self.eval_items[self._current_episode_num]
        self.evaluator = EpisodeEvaluator(
            item["episode_path"],
            strict_l2_instance_matching=self.strict_l2_instance_matching,
        )
        self.probe = item["probe"]
        self._current_step = 0
        self._cur_invalid_actions = 0
        self.episode_log = []
        self.model_actions = []
        self._last_result = None
        self._last_observation = None
        self._task_object_id = None
        self._last_navigation_target = None
        self._initial_state_adjustments = []

        self._episode_output_dir = os.path.join(
            self.log_path,
            f"episode_{self._current_episode_num + 1:04d}_{self._safe_name(item['episode_id'])}",
            self._safe_name(self.probe["probe_id"]),
        )
        os.makedirs(self._episode_output_dir, exist_ok=True)
        from emem_bench.construction.scene_utils.simulator import TrajectoryGenerator

        try:
            if self.sim is None:
                self.sim = TrajectoryGenerator(
                    scene=self.probe["scene"],
                    output_dir=self._episode_output_dir,
                    width=self.width,
                    height=self.height,
                    headless=self.headless,
                    allow_implicit_navigation=not self.strict_actions,
                )
            else:
                self.sim.reset_episode(
                    scene=self.probe["scene"],
                    output_dir=self._episode_output_dir,
                )
            self._apply_probe_state_changes()
            self._prepare_probe_initial_state()
            self._prepare_l2_retrieval_start_pose()
            return self._remember_observation(
                self._make_observation(
                    prefix="probe_init", caption=self.probe["instruction"]
                )
            )
        except Exception as exc:
            self._open_renderer_circuit(exc)
            raise RendererUnavailableError(self._renderer_failure) from exc

    def close(self):
        if self.sim is not None:
            sim = self.sim
            self.sim = None
            try:
                sim.close()
            except BrokenPipeError:
                # A renderer failure is already reported through RendererUnavailableError.
                pass

    def _open_renderer_circuit(self, exc: BaseException) -> None:
        self._renderer_circuit_open = True
        self._renderer_failure = f"{type(exc).__name__}: {exc}"

    def step(self, action, reasoning: str = "") -> tuple[dict, float, bool, dict]:
        if self._renderer_circuit_open:
            raise RendererUnavailableError(
                f"Renderer circuit is open: {self._renderer_failure or 'unknown failure'}"
            )
        if self.sim is None or self.evaluator is None or self.probe is None:
            raise RuntimeError("Call reset() before step().")

        action_dict, action_id, action_desc = self._resolve_action(action)
        if action_dict is None:
            self._cur_invalid_actions += 1
            self._current_step += 1
            self.model_actions.append(
                {
                    "action_type": "Invalid",
                    "target": None,
                    "instrument": None,
                    "success": False,
                    "feedback": "invalid action",
                }
            )
            self._last_result = self.evaluator.evaluate_probe(
                self.probe,
                self.model_actions,
                world_state=self._world_state_snapshot(),
            )
            done = self._current_step >= self._episode_max_steps()
            obs = self._make_observation()
            info = self._info(
                action_id=action_id,
                action_description=action_desc,
                last_action_success=0.0,
                env_feedback="invalid action",
            )
            self.episode_log.append({**info, "reasoning": reasoning})
            return obs, -1.0, done, info

        if action_dict["action_type"] in {"Done", "Stop", "MemoryInsufficient"}:
            return self._stop_episode(action_dict, action_id, action_desc, reasoning)

        try:
            step = self.sim.execute_and_record(
                action_type=action_dict["action_type"],
                target=action_dict.get("target"),
                instrument=action_dict.get("instrument"),
                nl_description=action_dict.get("natural_language") or action_desc,
            )
        except Exception as exc:
            self._open_renderer_circuit(exc)
            raise RendererUnavailableError(self._renderer_failure) from exc
        success = bool(step.feedback.success)
        if not success:
            self._cur_invalid_actions += 1
        if success and action_dict["action_type"] == "Navigate":
            self._last_navigation_target = action_dict.get("target")

        feedback = step.feedback.message
        model_action = {
            "action_type": action_dict["action_type"],
            "target": action_dict.get("target"),
            "target_label": action_dict.get("target_label"),
            "target_type": action_dict.get("target_type"),
            "parent_target": action_dict.get("parent_target"),
            "parent_target_label": action_dict.get("parent_target_label"),
            "parent_target_type": action_dict.get("parent_target_type"),
            "instrument": action_dict.get("instrument"),
            "success": success,
            "feedback": feedback,
        }
        self.model_actions.append(model_action)
        self._current_step += 1
        self._last_result = self.evaluator.evaluate_probe(
            self.probe,
            self.model_actions,
            world_state=self._world_state_snapshot(),
        )
        critical_msg = self._critical_error_message()

        done = (
            self._last_result.task_completed
            or self._current_step >= self._episode_max_steps()
        )
        reward = self._reward(success=success, done=done)
        obs = self._remember_observation(self._observation_dict(step.image_path))
        if critical_msg:
            feedback = f"{feedback}. A critical memory-dependent error was triggered."
        feedback = self._label_text(feedback)
        info = self._info(
            action_id=action_id,
            action_description=action_desc,
            last_action_success=1.0 if success else 0.0,
            env_feedback=feedback,
        )
        self.episode_log.append(
            {
                **info,
                "reasoning": reasoning,
                "action": model_action,
                "image_path": obs.get("image_path"),
            }
        )
        return obs, reward, done, info

    def _stop_episode(
        self,
        action_dict: dict,
        action_id: int,
        action_desc: str,
        reasoning: str,
    ) -> tuple[dict, float, bool, dict]:
        model_action = {
            "action_type": action_dict["action_type"],
            "target": action_dict.get("target"),
            "instrument": action_dict.get("instrument"),
        }
        self.model_actions.append(model_action)
        self._current_step += 1
        self._last_result = self.evaluator.evaluate_probe(
            self.probe,
            self.model_actions,
            world_state=self._world_state_snapshot(),
        )
        completed = bool(self._last_result.task_completed)
        if action_dict["action_type"] == "MemoryInsufficient":
            feedback = (
                "Correctly declared memory insufficient"
                if completed
                else "Declared memory insufficient for an answerable or already-committed task"
            )
        else:
            feedback = (
                "Stopped after task completion"
                if completed
                else "Stopped before task completion"
            )
        obs = self._remember_observation(
            self._make_observation(prefix="probe_done", caption=feedback)
        )
        info = self._info(
            action_id=action_id,
            action_description=action_desc,
            last_action_success=1.0 if completed else 0.0,
            env_feedback=feedback,
        )
        self.episode_log.append(
            {
                **info,
                "reasoning": reasoning,
                "action": model_action,
                "image_path": obs.get("image_path"),
            }
        )
        return obs, self._reward(success=completed, done=True), True, info

    # ─── Logging / scoring ─────────────────────────────────

    def current_result(self):
        if self.evaluator is None or self.probe is None:
            return None
        if self._last_result is None:
            self._last_result = self.evaluator.evaluate_probe(
                self.probe,
                self.model_actions,
                world_state=self._world_state_snapshot(),
            )
        return self._last_result

    def save_episode_log(self):
        os.makedirs(self._episode_output_dir, exist_ok=True)
        result = self.current_result()
        payload = {
            "episode_index": self._current_episode_num + 1,
            "instruction": self.probe["instruction"] if self.probe else "",
            "initial_state_adjustments": self._initial_state_adjustments,
            "model_actions": self.model_actions,
            "episode_log": self.episode_log,
            "result": asdict(result) if result else None,
        }
        with open(
            os.path.join(self._episode_output_dir, "episode_log.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        self._current_episode_num += 1

    def skip_current_episode(self) -> None:
        """Advance over an already completed episode without touching the renderer."""
        if self._current_episode_num >= self.number_of_episodes:
            raise StopIteration("No more evaluation episodes.")
        self._current_episode_num += 1

    # ─── Observation / action space ────────────────────────

    def _make_observation(self, prefix: str = "probe_obs", caption: str = "") -> dict:
        if self.sim is None:
            raise RuntimeError("Environment is not reset.")
        filename = self.sim.save_observation(prefix=prefix, caption=caption)
        return self._observation_dict(filename)

    def _observation_dict(self, image_filename: str) -> dict:
        self._refresh_action_space()
        visible_objects = []
        for obj in self.sim.make_visible_objects_list():
            obj_dict = self._to_dict(obj)
            obj_dict["object_label"] = self.object_labels.get(
                obj_dict["object_id"],
                obj_dict["object_type"],
            )
            visible_objects.append(obj_dict)
        observation = {
            "image_path": os.path.join(self._episode_output_dir, image_filename),
            "image_filename": image_filename,
            "episode_id": self.evaluator.episode_id if self.evaluator else None,
            "instruction": self.probe["instruction"] if self.probe else "",
            "scene": self.probe.get("scene") if self.probe else None,
            "visible_objects": visible_objects,
            "agent_state": self._to_dict(self.sim.get_agent_state()),
            "available_actions": self.available_actions,
            "object_labels": self.object_labels,
            "context": self.evaluator.get_context_for_model() if self.evaluator else [],
        }
        if self.expose_l2_task_object_label and self._is_l2_probe():
            target_id = self._l2_expected_pickup_target()
            if target_id:
                observation["task_object_label"] = self.object_labels.get(
                    target_id
                ) or object_type(target_id)
        return observation

    def _remember_observation(self, obs: dict) -> dict:
        self._last_observation = obs
        return obs

    def _minimal_observation(self) -> dict:
        if self._last_observation is not None:
            obs = dict(self._last_observation)
            obs["available_actions"] = []
            return obs
        observation = {
            "image_path": None,
            "image_filename": None,
            "episode_id": self.evaluator.episode_id if self.evaluator else None,
            "instruction": self.probe["instruction"] if self.probe else "",
            "visible_objects": [],
            "agent_state": {},
            "available_actions": [],
            "object_labels": self.object_labels,
            "context": self.evaluator.get_context_for_model() if self.evaluator else [],
        }
        if self.expose_l2_task_object_label and self._is_l2_probe():
            target_id = self._l2_expected_pickup_target()
            if target_id:
                observation["task_object_label"] = self.object_labels.get(
                    target_id
                ) or object_type(target_id)
        return observation

    def _prepare_probe_initial_state(self):
        if self.sim is None or self.evaluator is None or self.probe is None:
            return

        task_type = self._task_object_type_from_instruction()
        if not task_type:
            return
        self._task_object_id = self._probe_task_object_id(task_type)
        if not self._task_object_id:
            return

        for rule in self.evaluator.get_hidden_rules():
            condition = rule.get("condition") or {}
            if condition.get("rule_family") != "owner_habit":
                continue
            preferred_target = condition.get("preferred_target")
            if not preferred_target:
                continue
            if task_type not in split_type_list(condition.get("objectType")):
                continue

            parent_id = self.sim._find_object_current_parent(self._task_object_id)
            parent_type = object_type(parent_id)
            needs_relocation = (
                self._owner_habit_target_matches(parent_type, preferred_target)
                or not self._task_object_ready_for_pickup()
            )
            if not needs_relocation:
                return

            dest_id = self.sim.relocate_object_out_of_receptacle_type(
                self._task_object_id,
                preferred_target,
            )
            if dest_id:
                self._initial_state_adjustments.append(
                    {
                        "reason": (
                            "task_object_started_in_goal_receptacle"
                            if self._owner_habit_target_matches(
                                parent_type, preferred_target
                            )
                            else "task_object_started_unreachable"
                        ),
                        "object_id": self._task_object_id,
                        "from_parent": parent_id,
                        "to_parent": dest_id,
                    }
                )
            return

    def _owner_habit_target_matches(
        self, actual_type: str, preferred_target: str
    ) -> bool:
        if not actual_type:
            return False
        if actual_type == preferred_target:
            return True
        return actual_type in self._expected_put_target_types()

    def _expected_put_target_types(self) -> set[str]:
        if not self.probe:
            return set()
        target_types: set[str] = set()
        for action in self.probe.get("expected_actions", []):
            if action.get("action_type") == "PutObject" and action.get("target"):
                target_types.add(object_type(action.get("target")))
        for subtask in self.probe.get("sub_tasks", []):
            for action in subtask.get("expected_actions", []):
                if action.get("action_type") == "PutObject" and action.get("target"):
                    target_types.add(object_type(action.get("target")))
        return target_types

    def _prepare_l2_retrieval_start_pose(self):
        if self.sim is None or not self._is_l2_retrieval_probe():
            return

        target_id = self._l2_expected_pickup_target()
        if not target_id:
            return
        target_obj = self.sim._get_obj_meta(target_id)
        if not target_obj:
            return

        required_location = self._l2_required_location_target()
        target_pos = target_obj.get("position") or {}
        candidates = []
        for obj in self.sim.get_metadata()["objects"]:
            object_id = obj.get("objectId")
            if not object_id or object_id in {target_id, required_location}:
                continue
            if obj.get("pickupable"):
                continue
            if not (obj.get("receptacle") or obj.get("openable")):
                continue
            pos = obj.get("position") or {}
            dx = float(pos.get("x", 0.0)) - float(target_pos.get("x", 0.0))
            dz = float(pos.get("z", 0.0)) - float(target_pos.get("z", 0.0))
            candidates.append((-(dx * dx + dz * dz), object_id))

        for _, object_id in sorted(candidates):
            if not self.sim.navigate_to_object(object_id):
                continue
            if self._turn_until_target_hidden(target_id):
                self._initial_state_adjustments.append(
                    {
                        "reason": "l2_memory_probe_start_pose",
                        "object_id": target_id,
                        "start_near": object_id,
                        "required_location": required_location,
                    }
                )
                return

        self._turn_until_target_hidden(target_id)

    def _turn_until_target_hidden(self, target_id: str) -> bool:
        if self.sim is None:
            return False
        for _ in range(4):
            target_obj = self.sim._get_obj_meta(target_id)
            if target_obj and not target_obj.get("visible"):
                return True
            self.sim.controller.step(action="RotateRight")
        target_obj = self.sim._get_obj_meta(target_id)
        return bool(target_obj and not target_obj.get("visible"))

    def _apply_probe_state_changes(self):
        if self.sim is None or self.probe is None:
            return
        for change in self.probe.get("state_changes", []):
            if change.get("change_type") != "position_change":
                continue
            object_id = change.get("object_id")
            target_id = change.get("new_value")
            if not object_id or not target_id:
                continue
            success = self._place_object_for_probe_setup(object_id, target_id)
            self._initial_state_adjustments.append(
                {
                    "reason": "probe_state_change",
                    "object_id": object_id,
                    "to_parent": target_id,
                    "success": success,
                    "description": change.get("description", ""),
                }
            )

    def _place_object_for_probe_setup(self, object_id: str, target_id: str) -> bool:
        if self.sim is None:
            return False
        obj = self.sim._get_obj_meta(object_id)
        target = self.sim._get_obj_meta(target_id)
        if not obj or not target:
            return False

        current_parent_id = self.sim._find_object_current_parent(object_id)
        if current_parent_id == target_id:
            return True

        current_parent = (
            self.sim._get_obj_meta(current_parent_id) if current_parent_id else None
        )
        current_parent_was_open = bool(current_parent and current_parent.get("isOpen"))
        if (
            current_parent
            and current_parent.get("openable")
            and not current_parent.get("isOpen")
        ):
            self.sim.controller.step(
                action="OpenObject",
                objectId=current_parent_id,
                openness=1,
                forceAction=True,
            )

        self.sim.controller.step(
            action="PickupObject",
            objectId=object_id,
            forceAction=True,
            manualInteract=False,
        )
        if not self.sim.controller.last_event.metadata["lastActionSuccess"]:
            return False

        if target.get("openable") and not target.get("isOpen"):
            self.sim.controller.step(
                action="OpenObject",
                objectId=target_id,
                openness=1,
                forceAction=True,
            )
        self.sim.navigate_to_object(target_id)
        self.sim.controller.step(
            action="PutObject",
            objectId=target_id,
            forceAction=True,
            placeStationary=True,
        )
        success = bool(self.sim.controller.last_event.metadata["lastActionSuccess"])
        if success:
            if hasattr(self.sim, "_forget_drawer_attachment"):
                self.sim._forget_drawer_attachment(object_id)
            self.sim._object_parent_hints[object_id] = target_id
            if hasattr(self.sim, "_relocated_object_is_pickupable"):
                success = self.sim._relocated_object_is_pickupable(object_id, target_id)
            if (
                not success
                and hasattr(self.sim, "teleport_object_onto_receptacle_surface")
                and self.sim._can_surface_teleport_fallback(target)
            ):
                success = self.sim.teleport_object_onto_receptacle_surface(
                    object_id=object_id,
                    receptacle_id=target_id,
                    object_rot=obj.get("rotation"),
                    validate_pickup=True,
                )
        else:
            self.sim.controller.step(action="DropHandObject", forceAction=True)

        if (
            current_parent
            and current_parent.get("openable")
            and not current_parent_was_open
        ):
            self.sim.controller.step(
                action="CloseObject",
                objectId=current_parent_id,
                forceAction=True,
            )
        return success

    def _task_object_ready_for_pickup(self) -> bool:
        if self.sim is None or not self._task_object_id:
            return False
        obj = self.sim._get_obj_meta(self._task_object_id)
        if not obj or not obj.get("visible"):
            return False
        parent_id = self.sim._find_object_current_parent(self._task_object_id)
        parent_obj = self.sim._get_obj_meta(parent_id) if parent_id else None
        return not bool(
            parent_obj and parent_obj.get("openable") and not parent_obj.get("isOpen")
        )

    def _world_state_snapshot(self) -> dict:
        if self.sim is None:
            return {}
        objects = []
        try:
            metadata = self.sim.get_metadata()
        except Exception as exc:
            return {
                "task_object_id": self._task_object_id,
                "objects": [],
                "sim_error": f"{type(exc).__name__}: {exc}",
            }
        for obj in metadata["objects"]:
            object_id = obj.get("objectId")
            parent = None
            if object_id and hasattr(self.sim, "_find_object_current_parent"):
                try:
                    parent = self.sim._find_object_current_parent(object_id)
                except Exception:
                    parent = None
            objects.append(
                {
                    "object_id": object_id,
                    "object_type": obj.get("objectType"),
                    "parent": parent,
                    "visible": obj.get("visible"),
                    "is_open": obj.get("isOpen"),
                    "is_toggled": obj.get("isToggled"),
                    "is_picked_up": obj.get("isPickedUp"),
                    "receptacle_object_ids": obj.get("receptacleObjectIds") or [],
                    "parent_receptacles": obj.get("parentReceptacles") or [],
                }
            )
        return {
            "agent_state": self._safe_agent_state(),
            "task_object_id": self._task_object_id,
            "objects": objects,
        }

    def _safe_agent_state(self) -> dict:
        if self.sim is None:
            return {}
        try:
            return self._to_dict(self.sim.get_agent_state())
        except Exception:
            return {}

    def _task_object_type_from_instruction(self) -> Optional[str]:
        if not self.probe:
            return None
        match = re.search(
            r"\bthis\s+([A-Za-z][A-Za-z0-9_]*)\b",
            self.probe.get("instruction", ""),
            flags=re.I,
        )
        if not match:
            return None
        wanted = match.group(1).lower()
        if self.sim is None:
            return match.group(1)
        for obj in self.sim.get_metadata()["objects"]:
            obj_type = obj.get("objectType")
            if obj_type and obj_type.lower() == wanted:
                return obj_type
        return match.group(1)

    def _probe_task_object_id(self, task_type: str) -> Optional[str]:
        if self.sim is None or self.probe is None:
            return None
        for action in self.probe.get("expected_actions", []):
            target = action.get("target")
            if (
                action.get("action_type") == "PickUp"
                and object_type(target) == task_type
            ):
                return target
        for subtask in self.probe.get("sub_tasks", []):
            for action in subtask.get("expected_actions", []):
                target = action.get("target")
                if (
                    action.get("action_type") == "PickUp"
                    and object_type(target) == task_type
                ):
                    return target
        for obj in self.sim.get_metadata()["objects"]:
            if obj.get("objectType") == task_type and obj.get("pickupable"):
                return obj.get("objectId")
        return None

    def _is_l2_retrieval_probe(self) -> bool:
        return bool(
            self.probe
            and self.probe.get("probe_level") == "L2"
            and self.probe.get("probe_id") in {"probe_retrieve", "probe_track_location"}
        )

    def _is_l2_probe(self) -> bool:
        return bool(self.probe and self.probe.get("probe_level") == "L2")

    def _l2_expected_pickup_target(self) -> Optional[str]:
        if not self.probe:
            return None
        for action in self.probe.get("expected_actions", []):
            if action.get("action_type") == "PickUp":
                return action.get("target")
        for subtask in self.probe.get("sub_tasks", []):
            for action in subtask.get("expected_actions", []):
                if action.get("action_type") == "PickUp":
                    return action.get("target")
        return None

    def _l2_required_location_target(self) -> Optional[str]:
        if not self.probe:
            return None
        for action in self.probe.get("expected_actions", []):
            if action.get("action_type") == "Navigate":
                return action.get("target")
        for subtask in self.probe.get("sub_tasks", []):
            for action in subtask.get("expected_actions", []):
                if action.get("action_type") == "Navigate":
                    return action.get("target")
        return None

    def _l2_required_location_reached(self) -> bool:
        required_location = self._l2_required_location_target()
        if not required_location:
            return True
        return any(
            action.get("success") is not False
            and action.get("action_type") == "Navigate"
            and targets_match(action.get("target"), required_location, "Navigate")
            for action in self.model_actions
        )

    def _refresh_action_space(self):
        objects = self._candidate_objects()
        self.object_labels = self._object_labels(objects)
        holding = self.sim.get_agent_state().holding if self.sim is not None else None
        actions: list[dict] = []
        for obj in objects:
            object_id = obj["objectId"]
            object_type = obj["objectType"]
            label = self.object_labels.get(object_id, object_type)
            parent_fields = self._object_parent_action_fields(obj)
            if self._should_expose_navigation(obj):
                actions.append(
                    {
                        **self._action_entry(
                            "Navigate",
                            object_id,
                            f"Navigate to {label}",
                            label,
                            object_type,
                        ),
                        **parent_fields,
                    }
                )
            if (
                not holding
                and obj.get("pickupable")
                and self._should_expose_pickup(obj)
            ):
                actions.append(
                    {
                        **self._action_entry(
                            "PickUp", object_id, f"Pick up {label}", label, object_type
                        ),
                        **parent_fields,
                    }
                )
            receptacle_is_currently_placeable = not (
                obj.get("openable") and not obj.get("isOpen")
            )
            if (
                holding
                and obj.get("receptacle")
                and receptacle_is_currently_placeable
                and self._should_expose_object_interactions(obj)
            ):
                actions.append(
                    {
                        **self._action_entry(
                            "PutObject",
                            object_id,
                            f"Put held object on/in {label}",
                            label,
                            object_type,
                        ),
                        **parent_fields,
                    }
                )
            if (
                holding
                and obj.get("receptacle")
                and receptacle_is_currently_placeable
                and self._should_expose_object_interactions(obj)
            ):
                actions.append(
                    {
                        **self._action_entry(
                            "TransferContents",
                            object_id,
                            f"Transfer contents to {label}",
                            label,
                            object_type,
                        ),
                        **parent_fields,
                    }
                )
            if obj.get("openable") and self._should_expose_object_interactions(obj):
                actions.append(
                    {
                        **self._action_entry(
                            "Open", object_id, f"Open {label}", label, object_type
                        ),
                        **parent_fields,
                    }
                )
                actions.append(
                    {
                        **self._action_entry(
                            "Close", object_id, f"Close {label}", label, object_type
                        ),
                        **parent_fields,
                    }
                )
            if obj.get("toggleable") and self._should_expose_object_interactions(obj):
                actions.append(
                    {
                        **self._action_entry(
                            "ToggleOn",
                            object_id,
                            f"Toggle on {label}",
                            label,
                            object_type,
                        ),
                        **parent_fields,
                    }
                )
                actions.append(
                    {
                        **self._action_entry(
                            "ToggleOff",
                            object_id,
                            f"Toggle off {label}",
                            label,
                            object_type,
                        ),
                        **parent_fields,
                    }
                )

        for action_type in ("RotateLeft", "RotateRight", "MoveForward"):
            actions.append(self._action_entry(action_type, None, action_type))
        if self.probe and self.probe.get("allow_memory_insufficient") is True:
            actions.append(
                self._action_entry(
                    "MemoryInsufficient",
                    None,
                    "Memory insufficient — abstain instead of guessing a hidden target",
                )
            )
        actions.append(self._action_entry("Done", None, "Done / stop current task"))

        for idx, action in enumerate(actions):
            action["action_id"] = idx
        self.available_actions = actions

    def _candidate_objects(self) -> list[dict]:
        if self.sim is None:
            return []
        objects = self.sim.get_metadata()["objects"]
        by_id = {o["objectId"]: o for o in objects}
        visible_ids = {o["objectId"] for o in self.sim.get_visible_objects()}
        candidate_ids = set(visible_ids)
        candidate_ids.update(o["objectId"] for o in objects if o.get("pickupable"))

        if self.include_all_targets:
            for obj in objects:
                if (
                    obj.get("receptacle")
                    or obj.get("openable")
                    or obj.get("toggleable")
                ):
                    candidate_ids.add(obj["objectId"])

        return sorted(
            [
                by_id[oid]
                for oid in candidate_ids
                if oid in by_id
                and by_id[oid].get("objectType") not in STRUCTURAL_ACTION_EXCLUDE_TYPES
                and not by_id[oid].get("isPickedUp")
            ],
            key=lambda o: (o.get("objectType", ""), o.get("objectId", "")),
        )

    def _should_expose_navigation(self, obj: dict) -> bool:
        if self._is_l2_retrieval_probe() and obj.get("pickupable"):
            return (
                self.expose_pickupable_navigation
                and self._l2_required_location_reached()
            )
        return self.expose_pickupable_navigation or not bool(obj.get("pickupable"))

    def _should_expose_pickup(self, obj: dict) -> bool:
        if not obj.get("pickupable"):
            return False
        recently_navigated = obj.get("objectId") == self._last_navigation_target
        if not obj.get("visible") and not recently_navigated:
            return False
        if (
            self._is_l2_retrieval_probe()
            and not self._l2_required_location_reached()
            and not recently_navigated
        ):
            return False
        if self.sim is None:
            return True
        parent_id = self.sim._find_object_current_parent(obj["objectId"])
        parent_obj = self.sim._get_obj_meta(parent_id) if parent_id else None
        return not bool(
            parent_obj and parent_obj.get("openable") and not parent_obj.get("isOpen")
        )

    def _should_expose_object_interactions(self, obj: dict) -> bool:
        if self.strict_actions:
            return (
                bool(obj.get("visible"))
                or obj.get("objectId") == self._last_navigation_target
            )
        return bool(obj.get("visible")) or not bool(obj.get("pickupable"))

    def _object_parent_action_fields(self, obj: dict) -> dict:
        if self.sim is None:
            return {}
        object_id = obj.get("objectId")
        parent_id = (
            self.sim._find_object_current_parent(object_id) if object_id else None
        )
        if not parent_id or parent_id == object_id:
            return {}
        parent_obj = self.sim._get_obj_meta(parent_id)
        parent_type = (
            parent_obj.get("objectType") if parent_obj else object_type(parent_id)
        )
        parent_label = self.object_labels.get(parent_id, parent_type)
        return {
            "parent_target": parent_id,
            "parent_target_label": parent_label,
            "parent_target_type": parent_type,
        }

    @staticmethod
    def _object_labels(objects: list[dict]) -> dict[str, str]:
        by_type: dict[str, list[dict]] = {}
        for obj in objects:
            by_type.setdefault(obj.get("objectType", "Object"), []).append(obj)

        labels: dict[str, str] = {}
        for object_type, typed_objects in by_type.items():
            typed_objects = sorted(typed_objects, key=lambda o: o.get("objectId", ""))
            if len(typed_objects) == 1:
                labels[typed_objects[0]["objectId"]] = object_type
                continue
            for idx, obj in enumerate(typed_objects, start=1):
                labels[obj["objectId"]] = f"{object_type}_{idx}"
        return labels

    @staticmethod
    def _action_entry(
        action_type: str,
        target: Optional[str],
        text: str,
        target_label: Optional[str] = None,
        target_type: Optional[str] = None,
    ) -> dict:
        entry = {
            "action_id": -1,
            "action_type": action_type,
            "target": target,
            "target_label": target_label,
            "target_type": target_type,
            "text": text,
        }
        return entry

    def _resolve_action(self, action) -> tuple[Optional[dict], int, str]:
        if isinstance(action, int):
            if action < 0 or action >= len(self.available_actions):
                return None, action, "invalid action"
            entry = dict(self.available_actions[action])
            return entry, action, entry["text"]

        if isinstance(action, str):
            if action.strip().lower() in {"done", "stop"}:
                action = {"action_type": "Done"}
            else:
                try:
                    action = json.loads(action)
                except json.JSONDecodeError:
                    return None, -1, action

        if not isinstance(action, dict):
            return None, -1, "invalid action"

        action_type = action.get("action_type") or action.get("action")
        if not action_type and "action_id" in action:
            try:
                return self._resolve_action(int(action["action_id"]))
            except (TypeError, ValueError):
                return None, -1, "invalid action_id"
        if action_type not in DEFAULT_ACTION_TYPES:
            return None, -1, f"unknown action: {action_type}"
        try:
            action_id = int(action.get("action_id", -1))
        except (TypeError, ValueError):
            action_id = -1
        action_dict = {
            "action_type": action_type,
            "target": action.get("target") or action.get("objectId"),
            "target_label": action.get("target_label"),
            "target_type": action.get("target_type"),
            "parent_target": action.get("parent_target"),
            "parent_target_label": action.get("parent_target_label"),
            "parent_target_type": action.get("parent_target_type"),
            "instrument": action.get("instrument"),
            "natural_language": action.get("natural_language")
            or action.get("text")
            or "",
        }
        desc = action_dict["natural_language"] or self._describe_action(action_dict)
        return action_dict, action_id, desc

    @staticmethod
    def _describe_action(action: dict) -> str:
        target = action.get("target")
        if target:
            return f"{action['action_type']} {target}"
        return action["action_type"]

    def _label_text(self, text: str) -> str:
        for object_id, label in sorted(
            self.object_labels.items(), key=lambda item: len(item[0]), reverse=True
        ):
            text = text.replace(object_id, label)
        return text

    # ─── Info helpers ──────────────────────────────────────

    def _info(
        self,
        *,
        action_id: int,
        action_description: str,
        last_action_success: float,
        env_feedback: str,
    ) -> dict:
        result = self.current_result()
        progress = 0.0
        task_success = 0
        if result and result.sub_task_results:
            task_success = int(result.task_completed)
            progress = result.task_progress
        return {
            "task_success": task_success,
            "task_progress": progress,
            "subgoal_reward": result.mae if result else 0.0,
            "env_step": self._current_step,
            "last_action_success": last_action_success,
            "action_id": action_id,
            "action_description": action_description,
            "env_feedback": env_feedback,
            "num_invalid_actions": self._cur_invalid_actions,
        }

    def _reward(self, *, success: bool, done: bool) -> float:
        result = self.current_result()
        if done and result:
            return float(result.mae)
        return 0.0 if success else -1.0

    def _critical_error_message(self) -> str:
        result = self.current_result()
        if not result:
            return ""
        latest_action_idx = len(self.model_actions) - 1
        for sub_result in result.sub_task_results:
            for event in sub_result.triggered_err:
                if event.step_index == latest_action_idx:
                    return "Critical memory error triggered."
        return ""

    def _episode_max_steps(self) -> int:
        if self._max_steps is not None:
            return self._max_steps
        if self.probe is None:
            return 60
        return int(self.probe.get("max_steps", 60))

    @staticmethod
    def _to_dict(value):
        if is_dataclass(value):
            return asdict(value)
        return value

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:120]

    @staticmethod
    def _collect_episode_paths(
        episode_path: str | None,
        episode_dir: str | None,
        episode_paths: list[str] | None = None,
    ) -> list[str]:
        if episode_paths is not None:
            return list(dict.fromkeys(str(Path(path)) for path in episode_paths))
        paths: list[str] = []
        if episode_path:
            paths.append(str(Path(episode_path)))
        if episode_dir:
            base = Path(episode_dir)
            for path in base.glob("*/*.json"):
                if path.parent.name.startswith("."):
                    continue
                if path.stem != path.parent.name:
                    continue
                paths.append(str(path))
        return sorted(set(paths))

    @staticmethod
    def _build_eval_items(episode_paths: list[str]) -> list[dict]:
        items = []
        for episode_path in episode_paths:
            evaluator = EpisodeEvaluator(episode_path)
            for probe in evaluator.get_probes():
                items.append(
                    {
                        "episode_path": episode_path,
                        "episode_id": evaluator.episode_id,
                        "probe": probe,
                    }
                )
        return items
