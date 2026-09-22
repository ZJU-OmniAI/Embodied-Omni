"""
Embodied Memorizer Bench - evaluation engine.

This module provides a practical v1 offline evaluator:
  1. Load one episode JSON.
  2. Expose context sessions for model input while keeping answers hidden.
  3. Score model action traces against micro/macro probes.
  4. Report SR, RAR, ERR, AES and MAE with human-readable trigger evidence.

Online AI2-THOR rollouts are intentionally left as a thin future layer so
the scoring logic remains easy to test with offline action logs.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from .matchers import (
    normalize_actions,
    object_type,
    split_type_list,
    target_slot_base,
    targets_match,
)
from .metrics import compute_aes, compute_mae, compute_sr


@dataclass
class PenaltyEvent:
    """One concrete RAR/ERR trigger hit."""

    step_index: int
    action_type: str
    target: Optional[str]
    penalty_type: str
    description: str
    trigger_condition: str = ""


@dataclass
class SubTaskResult:
    """Score and diagnostics for one probe sub-task."""

    sub_task_id: str
    instruction: str
    expected_actions: list[dict] = field(default_factory=list)
    matched_expected: list[dict] = field(default_factory=list)
    missing_expected: list[dict] = field(default_factory=list)
    matched_action_indices: list[int] = field(default_factory=list)
    actual_steps: int = 0
    optimal_steps: int = 0
    task_completed: bool = False
    task_progress: float = 0.0
    sr: float = 0.0
    rar: float = 0.0
    err: float = 0.0
    aes: float = 0.0
    mae: float = 0.0
    invalid_count: int = 0
    error_count: int = 0
    triggered_rar: list[PenaltyEvent] = field(default_factory=list)
    triggered_err: list[PenaltyEvent] = field(default_factory=list)
    failure_reason: Optional[str] = None


@dataclass
class ProbeResult:
    """Score for one micro/macro probe."""

    probe_id: str
    probe_level: str
    instruction: str
    model_actions: list[dict] = field(default_factory=list)
    sub_task_results: list[SubTaskResult] = field(default_factory=list)
    actual_steps: int = 0
    optimal_steps: int = 0
    task_completed: bool = False
    task_progress: float = 0.0
    sr: float = 0.0
    rar: float = 0.0
    err: float = 0.0
    aes: float = 0.0
    mae: float = 0.0
    triggered_rar: list[str] = field(default_factory=list)
    triggered_err: list[str] = field(default_factory=list)


@dataclass
class EpisodeResult:
    """Aggregated score for an episode."""

    episode_id: str
    episode_name: str
    difficulty: str
    probe_results: list[ProbeResult] = field(default_factory=list)
    avg_sr: float = 0.0
    avg_rar: float = 0.0
    avg_err: float = 0.0
    avg_aes: float = 0.0
    avg_mae: float = 0.0


class EpisodeEvaluator:
    """Evaluate model actions on one episode JSON."""

    def __init__(
        self,
        episode_path: str,
        *,
        allow_same_receptacle_type: bool = True,
        strict_l2_instance_matching: bool = False,
    ):
        with open(episode_path, "r", encoding="utf-8") as f:
            self.episode = json.load(f)
        self.episode_path = episode_path
        self.episode_id = self.episode["episode_id"]
        self.episode_name = self.episode["episode_name"]
        self.difficulty = self.episode["difficulty"]
        self.allow_same_receptacle_type = allow_same_receptacle_type
        self.strict_l2_instance_matching = strict_l2_instance_matching

    # ─── Data exposed to models ─────────────────────────────

    def get_context_for_model(self) -> list[dict]:
        """Return context sessions without hidden answers or scoring hooks."""
        contexts = []
        for session in self.episode.get("sessions", []):
            traj = session.get("context_trajectory", {})
            hard_context = session.get("hard_context") or {}
            contexts.append(
                {
                    "session_name": session.get("session_name"),
                    "scene": session.get("scene"),
                    "room_type": session.get("room_type"),
                    "memory_namespace": hard_context.get("source_episode_id")
                    or self.episode_id,
                    "description": traj.get("description"),
                    "total_steps": traj.get("total_steps"),
                    "steps": traj.get("steps", []),
                }
            )
        return contexts

    def get_probes(self) -> list[dict]:
        """Return all probes. Evaluator-only fields stay inside probe dicts."""
        probes = []
        for session in self.episode.get("sessions", []):
            if session.get("micro_probe"):
                probes.append(session["micro_probe"])
        if self.episode.get("macro_probe"):
            probes.append(self.episode["macro_probe"])
        return probes

    def get_memory_cues(self) -> list[dict]:
        return self.episode.get("memory_cues", [])

    def get_hidden_rules(self) -> list[dict]:
        return self.episode.get("hidden_rules", [])

    # ─── Probe normalization ────────────────────────────────

    def _probe_subtasks(self, probe: dict) -> list[dict]:
        if probe.get("sub_tasks"):
            return probe["sub_tasks"]
        return [
            {
                "sub_task_id": probe.get("probe_id", "probe"),
                "instruction": probe.get("instruction", ""),
                "expected_actions": probe.get("expected_actions", []),
                "evaluation_hooks": probe.get("evaluation_hooks", {}),
                "optimal_steps": probe.get("optimal_steps"),
            }
        ]

    def oracle_actions_for_probe(self, probe: dict) -> list[dict]:
        """Return the concatenated oracle actions for quick smoke tests."""
        if probe.get("answerability") == "missing_evidence":
            contract = probe.get("abstention_contract") or {}
            if contract.get("correct_action") == "MemoryInsufficient":
                return [{"action_type": "MemoryInsufficient"}]
        actions: list[dict] = []
        for subtask in self._probe_subtasks(probe):
            actions.extend(normalize_actions(subtask.get("expected_actions", [])))
        return actions

    @staticmethod
    def _world_object_type(
        object_id: Optional[str], world_state: Optional[dict]
    ) -> str:
        if object_id and world_state:
            for obj in world_state.get("objects", []):
                if obj.get("object_id") == object_id:
                    return obj.get("object_type") or object_type(object_id)
        return object_type(object_id)

    # ─── Evaluation ─────────────────────────────────────────

    def evaluate_probe(
        self,
        probe: dict,
        model_actions: list[dict],
        world_state: Optional[dict] = None,
    ) -> ProbeResult:
        actions = normalize_actions(model_actions)
        scoring = {
            "lambda_rar": 5.0,
            "alpha": 1.0,
            "beta": 2.0,
            **(probe.get("scoring") or {}),
        }

        sub_results = [
            self._evaluate_subtask(probe, subtask, actions, scoring, world_state)
            for subtask in self._probe_subtasks(probe)
        ]

        result = ProbeResult(
            probe_id=probe["probe_id"],
            probe_level=probe["probe_level"],
            instruction=probe["instruction"],
            model_actions=actions,
            sub_task_results=sub_results,
            actual_steps=len(actions),
            optimal_steps=sum(st.optimal_steps for st in sub_results),
            task_completed=all(st.task_completed for st in sub_results),
        )
        self._aggregate_probe(result)
        return result

    def _evaluate_subtask(
        self,
        probe: dict,
        subtask: dict,
        actions: list[dict],
        scoring: dict,
        world_state: Optional[dict],
    ) -> SubTaskResult:
        hooks = subtask.get("evaluation_hooks") or probe.get("evaluation_hooks") or {}
        rar_events = self._find_rar_events(
            actions,
            hooks.get("rar_triggers", []),
            probe=probe,
            subtask=subtask,
            world_state=world_state,
        )
        err_events = self._find_err_events(
            actions,
            hooks.get("err_triggers", []),
            probe=probe,
            subtask=subtask,
            world_state=world_state,
        )
        completed = self._semantic_completed(
            probe=probe,
            subtask=subtask,
            actions=actions,
            err_events=err_events,
            world_state=world_state,
        )
        progress = self._semantic_progress(
            probe=probe,
            subtask=subtask,
            actions=actions,
            world_state=world_state,
        )
        sr = compute_sr(completed)

        invalid_count = len(rar_events)
        error_count = len(err_events)
        actual_steps = len(actions)
        optimal_steps = (
            1
            if probe.get("answerability") == "missing_evidence"
            else (
                subtask.get("optimal_steps")
                or probe.get("optimal_steps")
                or self._default_optimal_steps()
            )
        )
        rar = invalid_count / actual_steps if actual_steps else 0.0
        # In current L3 probes, the key memory-dependent decision is singular:
        # avoid repeating the known bad interaction. Keep ERR interpretable.
        n_memory_dependent = max(1, len(probe.get("required_memory_integration", [])))
        err = min(1.0, error_count / n_memory_dependent)
        aes = compute_aes(
            sr,
            optimal_steps,
            actual_steps,
            invalid_count,
            lambda_rar=float(scoring.get("lambda_rar", 5.0)),
        )
        mae = compute_mae(
            sr,
            rar,
            err,
        )

        failure_reason = None
        if err_events:
            first_err = err_events[0]
            failure_reason = (
                f"Triggered ERR: {first_err.action_type}({first_err.target})"
            )
        elif not completed:
            failure_reason = "Semantic goal state was not satisfied"

        return SubTaskResult(
            sub_task_id=subtask.get("sub_task_id", probe.get("probe_id", "probe")),
            instruction=subtask.get("instruction", probe.get("instruction", "")),
            actual_steps=actual_steps,
            optimal_steps=optimal_steps,
            task_completed=completed,
            task_progress=round(progress, 4),
            sr=round(sr, 4),
            rar=round(rar, 4),
            err=round(err, 4),
            aes=round(aes, 4),
            mae=round(mae, 4),
            invalid_count=invalid_count,
            error_count=error_count,
            triggered_rar=rar_events,
            triggered_err=err_events,
            failure_reason=failure_reason,
        )

    # ─── Semantic success checks ───────────────────────────────

    def _semantic_completed(
        self,
        *,
        probe: dict,
        subtask: dict,
        actions: list[dict],
        err_events: list[PenaltyEvent],
        world_state: Optional[dict],
    ) -> bool:
        """Evaluate success from the semantic end state, not oracle paths."""
        abstention_completed = self._abstention_completed_state(probe, actions)
        if abstention_completed is not None:
            return abstention_completed
        if err_events:
            return False

        l2_completed = self._l2_completed_state(probe, subtask, actions, world_state)
        if l2_completed is not None:
            return l2_completed
        if self._owner_habit_completed_state(probe, subtask, actions, world_state):
            return True
        if self._physical_constraint_completed_state(
            probe, subtask, actions, world_state
        ):
            return True
        return False

    @staticmethod
    def _abstention_completed_state(
        probe: dict,
        actions: list[dict],
    ) -> Optional[bool]:
        if probe.get("answerability") != "missing_evidence":
            return None
        contract = probe.get("abstention_contract") or {}
        allowed_before = set(contract.get("allowed_before_abstention") or [])
        for index, action in enumerate(actions):
            if action.get("action_type") != "MemoryInsufficient":
                continue
            return all(
                previous.get("action_type") in allowed_before
                for previous in actions[:index]
            )
        return False

    def _owner_habit_completed_state(
        self,
        probe: dict,
        subtask: dict,
        actions: list[dict],
        world_state: Optional[dict],
    ) -> bool:
        if not world_state:
            return False

        for rule in self._owner_habit_rules():
            task_types = self._task_object_types(probe, subtask, rule["object_types"])
            if not task_types:
                continue
            held_id: Optional[str] = None
            held_type = ""
            for action in actions:
                if not self._action_succeeded(action):
                    continue
                at = action.get("action_type", "")
                target = action.get("target")
                if at == "PickUp":
                    picked_type = self._world_object_type(target, world_state)
                    if picked_type in task_types:
                        held_id = target
                        held_type = picked_type
                elif at == "PutObject" and held_id and held_type in task_types:
                    target_type = self._world_object_type(target, world_state)
                    if not self._owner_habit_target_matches(target_type, rule, subtask):
                        held_id = None
                        held_type = ""
                        continue
                    placed_obj = self._world_object(world_state, held_id)
                    if (
                        placed_obj
                        and not placed_obj.get("is_picked_up")
                        and self._owner_habit_target_matches(
                            self._world_object_type(
                                placed_obj.get("parent"), world_state
                            ),
                            rule,
                            subtask,
                        )
                    ):
                        return True
                    held_id = None
                    held_type = ""
        return False

    @staticmethod
    def _task_state_objects(world_state: dict, task_types: set[str]) -> list[dict]:
        task_object_id = world_state.get("task_object_id")
        objects = world_state.get("objects", [])
        if task_object_id:
            return [obj for obj in objects if obj.get("object_id") == task_object_id]
        return [obj for obj in objects if obj.get("object_type") in task_types]

    def _physical_constraint_completed_state(
        self,
        probe: dict,
        subtask: dict,
        actions: list[dict],
        world_state: Optional[dict],
    ) -> bool:
        for rule in self._physical_constraint_rules():
            task_types = self._task_object_types(probe, subtask)
            target_types = self._task_target_types(probe, rule["target_types"])
            if not task_types or not target_types:
                continue
            task_object_id = (world_state or {}).get("task_object_id")
            holding_task = False
            transfer_seen = False
            appliance_target: Optional[str] = None
            for action in actions:
                if not self._action_succeeded(action):
                    continue
                at = action.get("action_type", "")
                target = action.get("target")
                if at == "PickUp" and self._is_task_action_target(
                    target, task_object_id, task_types
                ):
                    holding_task = True
                elif at == "TransferContents" and holding_task:
                    transfer_seen = True
                    holding_task = False
                elif (
                    at == "PutObject"
                    and transfer_seen
                    and object_type(target) in target_types
                ):
                    appliance_target = target
                elif (
                    at == "ToggleOn"
                    and transfer_seen
                    and object_type(target) in target_types
                    and targets_match(
                        target,
                        appliance_target or target,
                        "ToggleOn",
                        allow_same_receptacle_type=self.allow_same_receptacle_type,
                    )
                ):
                    if self._is_toggled_on(world_state, target):
                        return True
        return False

    def _semantic_progress(
        self,
        *,
        probe: dict,
        subtask: dict,
        actions: list[dict],
        world_state: Optional[dict],
    ) -> float:
        """EmbodiedBench-style progress: satisfied semantic predicates / total.

        The predicates are task-level states or successful high-level effects,
        not a match against the generated oracle action sequence.
        """
        abstention_progress = self._abstention_completed_state(probe, actions)
        if abstention_progress is not None:
            return float(abstention_progress)

        l2_progress = self._l2_progress(probe, subtask, actions, world_state)
        if l2_progress is not None:
            return l2_progress

        owner_progress = self._owner_habit_progress(
            probe, subtask, actions, world_state
        )
        if owner_progress is not None:
            return owner_progress

        physical_progress = self._physical_constraint_progress(
            probe, subtask, actions, world_state
        )
        if physical_progress is not None:
            return physical_progress

        return 0.0

    def _l2_completed_state(
        self,
        probe: dict,
        subtask: dict,
        actions: list[dict],
        world_state: Optional[dict],
    ) -> Optional[bool]:
        probe_id = probe.get("probe_id", "")
        if probe.get("probe_level") != "L2":
            return None
        if not world_state:
            return False

        if probe_id in {"probe_retrieve", "probe_track_location"}:
            return bool(
                self._l2_pickup_satisfies_target(subtask, actions, world_state)
                and self._l2_required_location_reached(subtask, actions)
            )

        if probe_id == "probe_avoid_failure":
            stash_id = self._expected_pickup_target(subtask)
            stash_type = self._world_object_type(stash_id, world_state)
            picked_id = self._l2_failure_picked_object_id(
                actions, world_state, stash_type
            )
            stash_obj = self._world_object(world_state, picked_id)
            if not stash_obj or stash_obj.get("is_picked_up"):
                return False
            parent_id = stash_obj.get("parent")
            locked_id = self._locked_container_id()
            placed = any(
                self._action_succeeded(action)
                and action.get("action_type") == "PutObject"
                and self._is_valid_failure_storage(
                    world_state, action.get("target"), locked_id
                )
                for action in actions
            )
            return (
                bool(picked_id)
                and placed
                and self._is_valid_failure_storage(world_state, parent_id, locked_id)
            )

        return None

    def _l2_progress(
        self,
        probe: dict,
        subtask: dict,
        actions: list[dict],
        world_state: Optional[dict],
    ) -> Optional[float]:
        probe_id = probe.get("probe_id", "")
        if probe.get("probe_level") != "L2":
            return None
        if not world_state:
            return 0.0

        if probe_id in {"probe_retrieve", "probe_track_location"}:
            reached = self._l2_required_location_reached(subtask, actions)
            picked = self._l2_pickup_satisfies_target(subtask, actions, world_state)
            return self._predicate_fraction([reached, picked])

        if probe_id == "probe_avoid_failure":
            stash_id = self._expected_pickup_target(subtask)
            stash_type = self._world_object_type(stash_id, world_state)
            picked_id = self._l2_failure_picked_object_id(
                actions, world_state, stash_type
            )
            stash_obj = self._world_object(world_state, picked_id)
            parent_id = stash_obj.get("parent") if stash_obj else None
            locked_id = self._locked_container_id()
            picked = bool(picked_id)
            reached_storage = any(
                self._action_succeeded(action)
                and action.get("action_type") in {"Navigate", "Open", "PutObject"}
                and self._is_valid_failure_storage(
                    world_state, action.get("target"), locked_id
                )
                for action in actions
            )
            placed = bool(
                stash_obj
                and not stash_obj.get("is_picked_up")
                and self._is_valid_failure_storage(world_state, parent_id, locked_id)
            )
            return self._predicate_fraction([picked, reached_storage, placed])

        return None

    def _owner_habit_progress(
        self,
        probe: dict,
        subtask: dict,
        actions: list[dict],
        world_state: Optional[dict],
    ) -> Optional[float]:
        if not world_state:
            return None

        for rule in self._owner_habit_rules():
            task_types = self._task_object_types(probe, subtask, rule["object_types"])
            if not task_types:
                continue

            picked = any(
                self._action_succeeded(action)
                and action.get("action_type") == "PickUp"
                and self._world_object_type(action.get("target"), world_state)
                in task_types
                for action in actions
            )
            reached_preferred = any(
                self._action_succeeded(action)
                and action.get("action_type") in {"Navigate", "Open", "PutObject"}
                and self._owner_habit_target_matches(
                    self._world_object_type(action.get("target"), world_state),
                    rule,
                    subtask,
                )
                for action in actions
            )
            placed_correctly = self._owner_habit_completed_state(
                probe,
                subtask,
                actions,
                world_state,
            )
            return self._predicate_fraction(
                [picked, reached_preferred, placed_correctly]
            )

        return None

    def _physical_constraint_progress(
        self,
        probe: dict,
        subtask: dict,
        actions: list[dict],
        world_state: Optional[dict],
    ) -> Optional[float]:
        for rule in self._physical_constraint_rules():
            task_types = self._task_object_types(probe, subtask)
            target_types = self._task_target_types(probe, rule["target_types"])
            if not task_types or not target_types:
                continue

            task_object_id = (world_state or {}).get("task_object_id")
            picked = False
            transfer_seen = False
            heated_safely = False
            holding_task = False

            for action in actions:
                if not self._action_succeeded(action):
                    continue
                at = action.get("action_type", "")
                target = action.get("target")
                if at == "PickUp" and self._is_task_action_target(
                    target, task_object_id, task_types
                ):
                    picked = True
                    holding_task = True
                elif at == "TransferContents" and holding_task:
                    transfer_seen = True
                    holding_task = False
                elif (
                    at == "ToggleOn"
                    and transfer_seen
                    and object_type(target) in target_types
                    and self._is_toggled_on(world_state, target)
                ):
                    heated_safely = True

            return self._predicate_fraction([picked, transfer_seen, heated_safely])

        return None

    @staticmethod
    def _predicate_fraction(predicates: list[bool]) -> float:
        if not predicates:
            return 0.0
        return sum(1 for predicate in predicates if predicate) / len(predicates)

    @staticmethod
    def _expected_pickup_target(subtask: dict) -> Optional[str]:
        for action in normalize_actions(subtask.get("expected_actions", [])):
            if action.get("action_type") == "PickUp":
                return action.get("target")
        return None

    @staticmethod
    def _expected_first_nav_target(subtask: dict) -> Optional[str]:
        for action in normalize_actions(subtask.get("expected_actions", [])):
            if action.get("action_type") == "Navigate":
                return action.get("target")
        return None

    def _l2_required_location_reached(self, subtask: dict, actions: list[dict]) -> bool:
        required_location = self._expected_first_nav_target(subtask)
        if not required_location:
            return True
        strict_instance = bool(getattr(self, "strict_l2_instance_matching", False))
        return any(
            self._action_succeeded(action)
            and action.get("action_type") == "Navigate"
            and targets_match(
                action.get("target"),
                required_location,
                "Navigate",
                allow_same_receptacle_type=(
                    getattr(self, "allow_same_receptacle_type", True)
                    and not strict_instance
                ),
            )
            for action in actions
        )

    def _l2_pickup_satisfies_target(
        self,
        subtask: dict,
        actions: list[dict],
        world_state: Optional[dict],
    ) -> bool:
        """Check L2 retrieval success while handling indistinguishable siblings.

        Generated L2 probes sometimes contain several same-type pickupable objects
        under the same remembered receptacle. The memory cue tells the agent only
        the object type and location, so exact objectId is not identifiable. In
        that narrow case, accept a same-type object picked from the required
        remembered parent; keep exact objectId scoring everywhere else.
        """
        expected_target = self._expected_pickup_target(subtask)
        if not expected_target:
            return False
        target_obj = self._world_object(world_state, expected_target)
        if target_obj and target_obj.get("is_picked_up"):
            return True

        if getattr(self, "strict_l2_instance_matching", False):
            return any(
                self._action_succeeded(action)
                and action.get("action_type") == "PickUp"
                and action.get("target") == expected_target
                for action in actions
            )

        expected_type = self._world_object_type(expected_target, world_state)
        required_location = self._expected_first_nav_target(subtask)
        if not expected_type or not required_location:
            return False

        for action in actions:
            if (
                not self._action_succeeded(action)
                or action.get("action_type") != "PickUp"
            ):
                continue
            actual_target = action.get("target")
            if actual_target == expected_target:
                return True
            if self._world_object_type(actual_target, world_state) != expected_type:
                continue
            if self._action_parent_matches_required_location(action, required_location):
                return True
        return False

    def _l2_failure_picked_object_id(
        self,
        actions: list[dict],
        world_state: Optional[dict],
        expected_type: str,
    ) -> Optional[str]:
        for action in actions:
            if (
                not self._action_succeeded(action)
                or action.get("action_type") != "PickUp"
            ):
                continue
            target = action.get("target")
            if self._world_object_type(target, world_state) == expected_type:
                return target
        return None

    def _action_parent_matches_required_location(
        self,
        action: dict,
        required_location: str,
    ) -> bool:
        parent_target = action.get("parent_target")
        if parent_target and targets_match(
            parent_target, required_location, "Navigate"
        ):
            return True

        parent_label = action.get("parent_target_label")
        if parent_label and target_slot_base(parent_label) == target_slot_base(
            required_location
        ):
            return True

        parent_type = action.get("parent_target_type")
        return bool(
            parent_type
            and "|" not in str(required_location)
            and parent_type == object_type(required_location)
        )

    @staticmethod
    def _world_object(
        world_state: Optional[dict], object_id: Optional[str]
    ) -> Optional[dict]:
        if not world_state or not object_id:
            return None
        for obj in world_state.get("objects", []):
            if obj.get("object_id") == object_id:
                return obj
        return None

    def _locked_container_id(self) -> Optional[str]:
        for rule in self.get_hidden_rules():
            if rule.get("rule_id") == "rule_locked":
                condition = rule.get("condition") or {}
                return condition.get("object_id")
        return None

    def _is_valid_failure_storage(
        self,
        world_state: Optional[dict],
        parent_id: Optional[str],
        locked_id: Optional[str],
    ) -> bool:
        if not world_state or not parent_id or parent_id == locked_id:
            return False
        parent = self._world_object(world_state, parent_id)
        return bool(
            parent
            and parent.get("receptacle_object_ids") is not None
            and parent.get("object_type")
            in {
                "Cabinet",
                "Fridge",
                "Microwave",
                "Safe",
                "Dresser",
                "Drawer",
            }
        )

    def _is_closed_failure_storage(
        self,
        world_state: Optional[dict],
        parent_id: Optional[str],
        locked_id: Optional[str],
    ) -> bool:
        if not self._is_valid_failure_storage(world_state, parent_id, locked_id):
            return False
        parent = self._world_object(world_state, parent_id)
        return bool(parent and not parent.get("is_open"))

    @staticmethod
    def _is_task_action_target(
        target: Optional[str],
        task_object_id: Optional[str],
        task_types: set[str],
    ) -> bool:
        if task_object_id:
            return target == task_object_id
        return object_type(target) in task_types

    def _task_object_types(
        self,
        probe: dict,
        subtask: dict,
        allowed_types: Optional[set[str]] = None,
    ) -> set[str]:
        instruction = f"{probe.get('instruction', '')} {subtask.get('instruction', '')}"
        candidates = set(allowed_types or self._all_episode_object_types())
        explicit = re.search(
            r"\bthis\s+([A-Za-z][A-Za-z0-9_]*)\b", instruction, flags=re.I
        )
        if explicit:
            explicit_type = explicit.group(1)
            if allowed_types:
                for obj_type in candidates:
                    if obj_type.lower() == explicit_type.lower():
                        return {obj_type}
            else:
                return {explicit_type}
        mentioned = {
            obj_type
            for obj_type in candidates
            if re.search(rf"\b{re.escape(obj_type)}\b", instruction, flags=re.I)
        }
        if mentioned:
            return mentioned
        return set()

    def _task_target_types(self, probe: dict, candidate_types: set[str]) -> set[str]:
        instruction = probe.get("instruction", "")
        mentioned = {
            obj_type
            for obj_type in candidate_types
            if re.search(rf"\b{re.escape(obj_type)}\b", instruction, flags=re.I)
        }
        return mentioned or set(candidate_types)

    def _all_episode_object_types(self) -> set[str]:
        types: set[str] = set()
        for session in self.episode.get("sessions", []):
            traj = session.get("context_trajectory") or {}
            for step in traj.get("steps", []):
                action = step.get("action") or {}
                target_type = object_type(action.get("target"))
                if target_type:
                    types.add(target_type)
        return types

    @staticmethod
    def _is_toggled_on(world_state: Optional[dict], object_id: Optional[str]) -> bool:
        if not world_state or not object_id:
            return False
        target_type = object_type(object_id)
        for obj in world_state.get("objects", []):
            if (
                obj.get("object_id") == object_id
                or obj.get("object_type") == target_type
            ):
                if obj.get("is_toggled"):
                    return True
        return False

    @staticmethod
    def _action_succeeded(action: dict) -> bool:
        return action.get("success", True) is not False

    @staticmethod
    def _default_optimal_steps() -> int:
        return 4

    # ─── RAR / ERR detection ────────────────────────────────

    def _find_rar_events(
        self,
        actions: list[dict],
        rar_triggers: list[dict],
        *,
        probe: dict,
        subtask: dict,
        world_state: Optional[dict],
    ) -> list[PenaltyEvent]:
        events: list[PenaltyEvent] = []
        held_type = ""
        owner_rules = self._owner_habit_rules()

        for idx, action in enumerate(actions):
            at = action.get("action_type", "")
            target = action.get("target")
            tgt_type = self._world_object_type(target, world_state)
            succeeded = self._action_succeeded(action)

            if not succeeded:
                events.append(
                    self._penalty_event(
                        idx,
                        action,
                        "invalid_action",
                        action.get("feedback") or "Action failed in the environment",
                    )
                )

            if idx > 0:
                prev = actions[idx - 1]
                if at == prev.get("action_type") and target == prev.get("target"):
                    events.append(
                        self._penalty_event(
                            idx,
                            action,
                            "redundant_repeat",
                            "连续重复同一动作和目标",
                        )
                    )

            if idx >= 2 and at in {"Open", "Close"}:
                prev = actions[idx - 1]
                prev2 = actions[idx - 2]
                if (
                    prev.get("target") == target == prev2.get("target")
                    and prev.get("action_type") != at
                    and prev2.get("action_type") == at
                ):
                    events.append(
                        self._penalty_event(
                            idx,
                            action,
                            "state_reversal",
                            "短时间内反复开关同一对象",
                        )
                    )

            for trigger in rar_triggers:
                if self._explicit_trigger_matches(actions, idx, trigger):
                    events.append(
                        self._penalty_event(
                            idx,
                            action,
                            trigger.get("penalty_type", "rar"),
                            trigger.get("description", ""),
                            trigger.get("trigger_condition", ""),
                        )
                    )

            if self._l2_wrong_navigation_before_required(
                probe, subtask, actions, idx, require_type_mismatch=False
            ):
                required = self._expected_first_nav_target(subtask)
                events.append(
                    self._penalty_event(
                        idx,
                        action,
                        "blind_navigation",
                        (
                            f"Navigated to {object_type(target) or target} before the remembered "
                            f"location {object_type(required) or required}"
                        ),
                        "Navigate to wrong location before remembered location",
                    )
                )

            if at == "PickUp" and succeeded:
                held_type = self._world_object_type(target, world_state)
            elif at == "PutObject" and succeeded:
                held_type = ""
            elif at == "Navigate" and held_type and succeeded:
                for rule in owner_rules:
                    if (
                        held_type in rule["object_types"]
                        and tgt_type
                        and not self._owner_habit_target_matches(
                            tgt_type, rule, subtask
                        )
                    ):
                        events.append(
                            self._penalty_event(
                                idx,
                                action,
                                "blind_navigation",
                                f"拿着{held_type}时导航到非推荐位置{tgt_type}",
                            )
                        )
                        break

        return self._dedupe_events(events)

    def _find_err_events(
        self,
        actions: list[dict],
        err_triggers: list[dict],
        *,
        probe: dict,
        subtask: dict,
        world_state: Optional[dict],
    ) -> list[PenaltyEvent]:
        events: list[PenaltyEvent] = []
        transfer_seen = False
        held_id: Optional[str] = None
        held_type = ""

        for idx, action in enumerate(actions):
            at = action.get("action_type", "")
            target = action.get("target")
            succeeded = self._action_succeeded(action)

            for trigger in err_triggers:
                if succeeded and self._explicit_trigger_matches(
                    actions, idx, trigger, transfer_seen=transfer_seen, held_id=held_id
                ):
                    events.append(
                        self._penalty_event(
                            idx,
                            action,
                            trigger.get("penalty_type", "critical_err"),
                            trigger.get("description", ""),
                            trigger.get("trigger_condition", ""),
                        )
                    )

            if self._l2_wrong_navigation_before_required(
                probe, subtask, actions, idx, require_type_mismatch=True
            ):
                required = self._expected_first_nav_target(subtask)
                events.append(
                    self._penalty_event(
                        idx,
                        action,
                        "critical_err",
                        (
                            f"Navigated to {object_type(target) or target} before the remembered "
                            f"location {object_type(required) or required}"
                        ),
                        "Navigate to wrong location before remembered location",
                    )
                )

            if at == "PutObject" and held_type and succeeded:
                for rule in self._owner_habit_rules():
                    if held_type in rule[
                        "object_types"
                    ] and not self._owner_habit_target_matches(
                        self._world_object_type(target, world_state),
                        rule,
                        subtask,
                    ):
                        events.append(
                            self._penalty_event(
                                idx,
                                action,
                                "critical_err",
                                f"未应用目标位置: {held_type} 应放到 {rule['preferred_target']}",
                                f"PutObject target must be {rule['preferred_target']}",
                            )
                        )
                        break

            if at == "PickUp" and succeeded:
                held_id = target
                held_type = self._world_object_type(target, world_state)
            elif at == "TransferContents" and succeeded:
                transfer_seen = True
            elif at == "PutObject" and succeeded:
                held_id = None
                held_type = ""

        return self._dedupe_events(events)

    def _l2_wrong_navigation_before_required(
        self,
        probe: dict,
        subtask: dict,
        actions: list[dict],
        idx: int,
        *,
        require_type_mismatch: bool,
    ) -> bool:
        if probe.get("probe_level") != "L2":
            return False
        if probe.get("probe_id") not in {"probe_retrieve", "probe_track_location"}:
            return False

        action = actions[idx]
        if (
            not self._action_succeeded(action)
            or action.get("action_type") != "Navigate"
        ):
            return False

        required = self._expected_first_nav_target(subtask)
        target = action.get("target")
        if not required or not target or targets_match(target, required, "Navigate"):
            return False
        if self._is_physical_parent_of_required_basin(target, required):
            return False
        if require_type_mismatch and object_type(target) == object_type(required):
            return False

        return not any(
            self._action_succeeded(prev)
            and prev.get("action_type") == "Navigate"
            and targets_match(prev.get("target"), required, "Navigate")
            for prev in actions[:idx]
        )

    @staticmethod
    def _is_physical_parent_of_required_basin(target: str, required: str) -> bool:
        if not required.endswith(("|SinkBasin", "|BathtubBasin")):
            return False
        parent = required.rsplit("|", 1)[0]
        return str(target or "") == parent

    def _explicit_trigger_matches(
        self,
        actions: list[dict],
        idx: int,
        trigger: dict,
        *,
        transfer_seen: bool = False,
        held_id: Optional[str] = None,
    ) -> bool:
        action = actions[idx]
        at = action.get("action_type", "")
        target = action.get("target") or ""
        cond = trigger.get("trigger_condition", "")
        if not cond:
            return False

        if cond == "Navigate to room other than current scene":
            return at in {"LeaveRoom", "EnterRoom"}

        m = re.fullmatch(r"PutObject\((.+)\) before TransferContents", cond)
        if m:
            return at == "PutObject" and target == m.group(1) and not transfer_seen

        m = re.fullmatch(r"PutObject\((.+)\)", cond)
        if m:
            return at == "PutObject" and target == m.group(1)

        m = re.fullmatch(r"DirectUnsafeInteraction\(task=(.+), target=(.+)\)", cond)
        if m:
            task_id, target_id = m.group(1), m.group(2)
            direct_task_action = target == task_id and at in {
                "ToggleOn",
                "ToggleOff",
                "Open",
                "PutObject",
            }
            direct_target_put = (
                at == "PutObject"
                and target == target_id
                and held_id in {None, task_id}
                and not transfer_seen
            )
            return direct_task_action or direct_target_put

        return bool(target and target in cond and (at in cond or "(" not in cond))

    def _owner_habit_rules(self) -> list[dict]:
        rules = []
        for rule in self.get_hidden_rules():
            condition = rule.get("condition") or {}
            if condition.get("rule_family") != "owner_habit":
                continue
            preferred = condition.get("preferred_target")
            object_types = split_type_list(condition.get("objectType"))
            if preferred and object_types:
                rules.append(
                    {
                        "preferred_target": preferred,
                        "object_types": object_types,
                        "description": rule.get("description", ""),
                    }
                )
        return rules

    @staticmethod
    def _expected_put_target_types(subtask: dict) -> set[str]:
        return {
            object_type(action.get("target"))
            for action in normalize_actions(subtask.get("expected_actions", []))
            if action.get("action_type") == "PutObject" and action.get("target")
        }

    def _owner_habit_target_matches(
        self,
        actual_type: str,
        rule: dict,
        subtask: dict,
    ) -> bool:
        """Match semantic habit targets to ProcTHOR physical receptacle types.

        L3 owner-habit rules are written in semantic terms such as Cabinet or
        Shelf, while ProcTHOR sometimes exposes the executable target as a
        physical type such as ShelvingUnit or TVStand. The evaluator can use the
        hidden oracle action target type to normalize this scoring-side alias;
        it is never exposed to the planner.
        """
        if not actual_type:
            return False
        if actual_type == rule.get("preferred_target"):
            return True
        return actual_type in self._expected_put_target_types(subtask)

    def _physical_constraint_rules(self) -> list[dict]:
        rules = []
        for rule in self.get_hidden_rules():
            condition = rule.get("condition") or {}
            if condition.get("rule_family") != "physical_constraint":
                continue
            target_types = split_type_list(condition.get("objectType"))
            if target_types:
                rules.append(
                    {
                        "target_types": target_types,
                        "description": rule.get("description", ""),
                    }
                )
        return rules

    @staticmethod
    def _penalty_event(
        step_index: int,
        action: dict,
        penalty_type: str,
        description: str,
        trigger_condition: str = "",
    ) -> PenaltyEvent:
        return PenaltyEvent(
            step_index=step_index,
            action_type=action.get("action_type", ""),
            target=action.get("target"),
            penalty_type=str(penalty_type),
            description=description,
            trigger_condition=trigger_condition,
        )

    @staticmethod
    def _dedupe_events(events: list[PenaltyEvent]) -> list[PenaltyEvent]:
        seen = set()
        result = []
        for event in events:
            key = (
                event.step_index,
                event.action_type,
                event.target,
                event.penalty_type,
                event.description,
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(event)
        return result

    # ─── Entrypoints ────────────────────────────────────────

    def evaluate_offline(
        self,
        probe_actions: dict[str, list[dict]] | list[dict],
    ) -> EpisodeResult:
        """Evaluate provided model action traces.

        Args:
            probe_actions:
                Either {probe_id: [actions]} or a single action list. A list is
                applied to the only probe, which is convenient for L3 episodes.
        """
        probes = self.get_probes()
        action_map = self._normalize_probe_actions(probe_actions, probes)
        result = EpisodeResult(
            episode_id=self.episode_id,
            episode_name=self.episode_name,
            difficulty=self.difficulty,
        )
        for probe in probes:
            actions = action_map.get(probe["probe_id"], [])
            result.probe_results.append(self.evaluate_probe(probe, actions))
        self._aggregate_episode(result)
        return result

    def evaluate_oracle(self) -> EpisodeResult:
        """Evaluate each probe's generated expected actions as a smoke test."""
        return self.evaluate_offline(
            {
                probe["probe_id"]: self.oracle_actions_for_probe(probe)
                for probe in self.get_probes()
            }
        )

    def evaluate_online(
        self,
        model_fn: Callable,
        max_steps: int = 30,
    ) -> EpisodeResult:
        """Placeholder for future AI2-THOR rollout integration."""
        raise NotImplementedError(
            "Online evaluation is not implemented yet. "
            "Use evaluate_offline() with action logs first."
        )

    def _normalize_probe_actions(
        self,
        probe_actions: dict[str, list[dict]] | list[dict],
        probes: list[dict],
    ) -> dict[str, list[dict]]:
        if isinstance(probe_actions, list):
            if len(probes) != 1:
                raise ValueError(
                    "Action list input is only valid for a single-probe episode"
                )
            return {probes[0]["probe_id"]: probe_actions}
        return {
            probe_id: actions.get("actions", actions)
            if isinstance(actions, dict)
            else actions
            for probe_id, actions in (probe_actions or {}).items()
        }

    def _aggregate_probe(self, result: ProbeResult):
        if not result.sub_task_results:
            return
        n = len(result.sub_task_results)
        result.sr = round(sum(st.sr for st in result.sub_task_results) / n, 4)
        result.task_progress = round(
            sum(st.task_progress for st in result.sub_task_results) / n, 4
        )
        result.rar = round(sum(st.rar for st in result.sub_task_results) / n, 4)
        result.err = round(sum(st.err for st in result.sub_task_results) / n, 4)
        result.aes = round(sum(st.aes for st in result.sub_task_results) / n, 4)
        result.mae = round(compute_mae(result.sr, result.rar, result.err), 4)
        result.triggered_rar = [
            self._format_event(event)
            for st in result.sub_task_results
            for event in st.triggered_rar
        ]
        result.triggered_err = [
            self._format_event(event)
            for st in result.sub_task_results
            for event in st.triggered_err
        ]

    def _aggregate_episode(self, result: EpisodeResult):
        if not result.probe_results:
            return
        n = len(result.probe_results)
        result.avg_sr = round(sum(p.sr for p in result.probe_results) / n, 4)
        result.avg_rar = round(sum(p.rar for p in result.probe_results) / n, 4)
        result.avg_err = round(sum(p.err for p in result.probe_results) / n, 4)
        result.avg_aes = round(sum(p.aes for p in result.probe_results) / n, 4)
        result.avg_mae = round(
            compute_mae(result.avg_sr, result.avg_rar, result.avg_err), 4
        )

    @staticmethod
    def _format_event(event: PenaltyEvent) -> str:
        return (
            f"step[{event.step_index}] {event.action_type}({event.target})"
            f" -> {event.description or event.penalty_type}"
        )

    # ─── Output ─────────────────────────────────────────────

    def save_result(self, result: EpisodeResult, output_path: str):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)

    def print_result(self, result: EpisodeResult):
        print(f"\n{'=' * 60}")
        print(f"Episode: {result.episode_name} ({result.difficulty})")
        print(f"{'=' * 60}")

        for pr in result.probe_results:
            status = "PASS" if pr.task_completed else "FAIL"
            print(f"\n  Probe [{pr.probe_level}] {pr.probe_id}: {status}")
            print(f"    指令: {pr.instruction}")
            print(f"    步数: {pr.actual_steps} (最优: {pr.optimal_steps})")
            print(
                f"    SR={pr.sr:.2f}  Progress={pr.task_progress:.2f}  RAR={pr.rar:.2f}  ERR={pr.err:.2f}"
                f"  AES={pr.aes:.2f}  MAE={pr.mae:.2f}"
            )

            for st in pr.sub_task_results:
                st_status = "PASS" if st.task_completed else "FAIL"
                print(
                    f"    - SubTask {st.sub_task_id}: {st_status}, "
                    f"progress={st.task_progress:.2f}, RAR={st.rar:.2f}, "
                    f"ERR={st.err:.2f}, MAE={st.mae:.2f}"
                )
                if st.failure_reason:
                    print(f"      reason: {st.failure_reason}")

            for t in pr.triggered_rar:
                print(f"    [RAR] {t}")
            for t in pr.triggered_err:
                print(f"    [ERR] {t}")

        print(f"\n  {'-' * 40}")
        print(
            f"  聚合: SR={result.avg_sr:.2f}  RAR={result.avg_rar:.2f}  "
            f"ERR={result.avg_err:.2f}  AES={result.avg_aes:.2f}  "
            f"MAE={result.avg_mae:.2f}"
        )
