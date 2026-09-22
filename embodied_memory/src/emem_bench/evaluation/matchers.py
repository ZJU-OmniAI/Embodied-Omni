"""
Lightweight action matching helpers for the evaluation engine.

The evaluator intentionally keeps matching simple and inspectable:
expected actions must appear as an ordered subsequence of model actions.
For receptacle targets we optionally allow same object type matching so a
reasonable alternative Drawer/Cabinet is not rejected only because its
objectId differs from the generated oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Optional


RECEPTACLE_TYPES = {
    "Cabinet",
    "CounterTop",
    "Desk",
    "DiningTable",
    "Drawer",
    "Fridge",
    "GarbageCan",
    "Microwave",
    "Shelf",
    "SideTable",
    "BathtubBasin",
    "Sink",
    "SinkBasin",
    "StoveBurner",
    "Toaster",
}


@dataclass
class SequenceMatch:
    completed: bool
    matched_expected: list[dict] = field(default_factory=list)
    missing_expected: list[dict] = field(default_factory=list)
    matched_action_indices: list[int] = field(default_factory=list)


def normalize_action(action: dict) -> dict:
    """Normalize common action dict variants to {action_type, target, ...}."""
    if not action:
        return {"action_type": "", "target": None}
    normalized = dict(action)
    normalized["action_type"] = (
        normalized.get("action_type")
        or normalized.get("action")
        or normalized.get("type")
        or ""
    )
    normalized["target"] = normalized.get("target") or normalized.get("objectId")
    return normalized


def normalize_actions(actions: list[dict]) -> list[dict]:
    return [normalize_action(a) for a in actions or []]


def object_type(object_id: Optional[str]) -> str:
    if not object_id:
        return ""
    parts = str(object_id).split("|")
    if parts and parts[-1] in {"BathtubBasin", "SinkBasin"}:
        return parts[-1]
    return parts[0] if parts else ""


def split_type_list(value: Optional[str]) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in str(value).split("|") if part.strip()}


def is_receptacle_target(object_id: Optional[str]) -> bool:
    return object_type(object_id) in RECEPTACLE_TYPES


def target_slot_base(object_id: Optional[str]) -> str:
    """Normalize ProcTHOR surface-slot ids back to the physical receptacle id."""
    return re.sub(r"___\d+$", "", str(object_id or ""))


def targets_match(
    actual_target: Optional[str],
    expected_target: Optional[str],
    action_type: str,
    allow_same_receptacle_type: bool = True,
) -> bool:
    if not expected_target:
        return True
    if actual_target == expected_target:
        return True
    if actual_target and target_slot_base(actual_target) == target_slot_base(
        expected_target
    ):
        return True
    if not allow_same_receptacle_type:
        return False
    if action_type not in {
        "Navigate",
        "PutObject",
        "Open",
        "Close",
        "TransferContents",
    }:
        return False
    return (
        is_receptacle_target(expected_target)
        and is_receptacle_target(actual_target)
        and object_type(actual_target) == object_type(expected_target)
    )


def actions_match(
    actual: dict,
    expected: dict,
    allow_same_receptacle_type: bool = True,
) -> bool:
    actual = normalize_action(actual)
    expected = normalize_action(expected)
    action_type = expected.get("action_type", "")
    if actual.get("action_type") != action_type:
        return False
    return targets_match(
        actual.get("target"),
        expected.get("target"),
        action_type,
        allow_same_receptacle_type=allow_same_receptacle_type,
    )


def match_ordered_subsequence(
    model_actions: list[dict],
    expected_actions: list[dict],
    allow_same_receptacle_type: bool = True,
) -> SequenceMatch:
    """Return ordered-subsequence completion details."""
    actions = normalize_actions(model_actions)
    expected = normalize_actions(expected_actions)
    if not expected:
        return SequenceMatch(completed=True)

    exp_idx = 0
    matched_expected: list[dict] = []
    matched_indices: list[int] = []
    for idx, action in enumerate(actions):
        if exp_idx >= len(expected):
            break
        exp = expected[exp_idx]
        if actions_match(
            action,
            exp,
            allow_same_receptacle_type=allow_same_receptacle_type,
        ):
            matched_expected.append(exp)
            matched_indices.append(idx)
            exp_idx += 1

    missing_expected = expected[exp_idx:]
    return SequenceMatch(
        completed=exp_idx >= len(expected),
        matched_expected=matched_expected,
        missing_expected=missing_expected,
        matched_action_indices=matched_indices,
    )
