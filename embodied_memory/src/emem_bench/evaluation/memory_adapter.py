"""Adapter that exposes the real /embodied_memorizer memory system to this benchmark.

The benchmark already has its own action planner and evaluator.  This adapter
keeps those fixed and swaps only the memory source: past context trajectories
are written into embodied_memorizer.MemorySystem, queried with the current probe
instruction, and the retrieved embodied_memorizer text is passed to the planner as
model-visible memory context.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .matchers import object_type
from .planner import (
    sanitize_context_action_text,
    sanitize_context_description,
    sanitize_context_feedback,
    strip_object_ids,
)


DEFAULT_MEMORY_MODULE_ROOT = "installed-package"
DEFAULT_LOCAL_EMBEDDING_MODEL = os.getenv("EMEM_EMBEDDING_MODEL", "lexical-hash")

CHILD_SLOT_CUE_BASE_TYPES = {
    "ShelvingUnit",
    "TVStand",
}

BASE_RECEPTACLE_CHILD_TYPES = {
    "Bathtub": "BathtubBasin",
    "Sink": "SinkBasin",
}

BASE_RECEPTACLE_CHILD_FAMILIES = {
    key.lower(): {value.lower()} for key, value in BASE_RECEPTACLE_CHILD_TYPES.items()
}


PLACEMENT_RE = re.compile(
    r"\b(?:Place|Put|Try placing|Move)\s+"
    r"(?:cleaned\s+)?(?P<object>[A-Za-z][A-Za-z0-9_]*)\s+"
    r"(?:casually\s+)?(?:in|into|on|onto|to)\s+"
    r"(?:the\s+)?(?P<receptacle>[A-Za-z][A-Za-z0-9_]*)",
    re.I,
)

CUE_LOCATION_RE = re.compile(
    r"放在\s*(?P<receptacle>[A-Za-z][A-Za-z0-9_]*)\s*[上里中内]?(?:面)?的\s*"
    r"(?P<object>[A-Za-z][A-Za-z0-9_]*)",
    re.I,
)
CUE_RELOCATION_RE = re.compile(
    r"把\s*(?P<object>[A-Za-z][A-Za-z0-9_]*)\s*从\s*"
    r"(?P<source>[A-Za-z][A-Za-z0-9_]*)\s*搬到[了到]?\s*"
    r"(?P<target>[A-Za-z][A-Za-z0-9_]*)",
    re.I,
)


def _ensure_memory_module(memory_module_root: str | None):
    from embodied_memorizer import MemorySystem, merge_config  # type: ignore
    from embodied_memorizer.config import DEFAULT_CONFIG  # type: ignore
    from embodied_memorizer.tools import _format_query_results  # type: ignore
    from embodied_memorizer.context import build_memory_context  # type: ignore

    return (
        MemorySystem,
        DEFAULT_CONFIG,
        merge_config,
        build_memory_context,
        _format_query_results,
    )


def _memory_config(
    *,
    default_config: dict,
    merge_config,
    embedding_model: str | None,
    embedding_device: str,
    max_memory_tokens: int,
    event_top_k: int,
    spatial_top_k: int,
    scene_top_k: int,
) -> dict:
    return merge_config(
        {
            **({"embedding_model": embedding_model} if embedding_model else {}),
            "embedding_device": embedding_device,
            "prompt": {
                "max_memory_tokens": max_memory_tokens,
                "event_top_k": event_top_k,
                "spatial_top_k": spatial_top_k,
                "scene_top_k": scene_top_k,
                "auto_event_record": False,
                "auto_spatial_update": False,
                "auto_scene_record": False,
            },
        },
        default_config,
    )


def _context_object_labels(contexts: list[dict]) -> dict[str, str]:
    by_type: dict[str, list[str]] = {}
    for session in contexts:
        steps = list(session.get("steps") or [])
        for step in steps:
            action = step.get("action") or {}
            target = action.get("target")
            if isinstance(target, str) and target:
                by_type.setdefault(object_type(target), []).append(target)
            for obj in step.get("visible_objects", []):
                object_id = obj.get("object_id") or obj.get("objectId")
                if isinstance(object_id, str) and object_id:
                    by_type.setdefault(object_type(object_id), []).append(object_id)

    labels: dict[str, str] = {}
    for obj_type, object_ids in by_type.items():
        unique_ids = sorted(set(object_ids))
        if len(unique_ids) == 1:
            labels[unique_ids[0]] = obj_type
        else:
            for idx, object_id in enumerate(unique_ids, start=1):
                labels[object_id] = f"{obj_type}_{idx}"
    return labels


def _object_label(
    object_id: Any,
    object_labels: dict[str, str] | None,
    context_labels: dict[str, str] | None,
) -> str:
    if not isinstance(object_id, str) or not object_id:
        return ""
    return (
        (object_labels or {}).get(object_id)
        or (context_labels or {}).get(object_id)
        or object_type(object_id)
    )


def _replace_ids_with_labels(
    text: Any,
    object_labels: dict[str, str] | None,
    context_labels: dict[str, str] | None,
    label_items: list[tuple[str, str]] | None = None,
) -> Any:
    if not isinstance(text, str):
        return text
    if label_items is None:
        labels = dict(context_labels or {})
        labels.update(object_labels or {})
        label_items = sorted(
            labels.items(), key=lambda item: len(item[0]), reverse=True
        )
    for object_id, label in label_items:
        text = text.replace(object_id, label)
    return text


def _merged_label_items(
    object_labels: dict[str, str] | None,
    context_labels: dict[str, str] | None,
) -> list[tuple[str, str]]:
    labels = dict(context_labels or {})
    labels.update(object_labels or {})
    return sorted(labels.items(), key=lambda item: len(item[0]), reverse=True)


def _visible_object_names(
    step: dict,
    object_labels: dict[str, str] | None,
    context_labels: dict[str, str] | None,
    limit: int = 16,
) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for obj in step.get("visible_objects", [])[:limit]:
        object_id = obj.get("object_id") or obj.get("objectId")
        name = _object_label(object_id, object_labels, context_labels)
        if not name:
            name = (
                obj.get("object_type")
                or obj.get("objectType")
                or object_type(object_id)
            )
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _action_text(
    step: dict,
    object_labels: dict[str, str] | None,
    context_labels: dict[str, str] | None,
    label_items: list[tuple[str, str]] | None = None,
    *,
    preserve_durable_failure: bool = False,
) -> tuple[str, bool, str]:
    action = step.get("action") or {}
    feedback = step.get("feedback") or {}
    action_type = action.get("action_type")
    target_label = _object_label(action.get("target"), object_labels, context_labels)
    text = _replace_ids_with_labels(
        action.get("natural_language") or action_type or "Unknown action",
        object_labels,
        context_labels,
        label_items,
    )
    text = sanitize_context_action_text(text, action_type)
    if target_label and target_label.lower() not in str(text).lower():
        text = f"{text} [target: {target_label}]"
    success = bool(feedback.get("success"))
    feedback_text = _replace_ids_with_labels(
        feedback.get("message") or "",
        object_labels,
        context_labels,
        label_items,
    )
    feedback_text = sanitize_context_feedback(
        feedback_text,
        success,
        preserve_durable_failure=preserve_durable_failure,
    )
    return str(text), success, str(feedback_text or "")


def _record_visible_objects(
    memory,
    session_name: str,
    step: dict,
    object_labels: dict[str, str] | None,
    context_labels: dict[str, str] | None,
) -> None:
    for name in _visible_object_names(step, object_labels, context_labels):
        memory.add_spatial(
            name=name,
            node_type="object",
            relations=[{"target": session_name, "type": "seen_in"}],
            properties={"visible_in_context": True},
        )


def _record_target_object(
    memory,
    action: dict,
    session_name: str,
    object_labels: dict[str, str] | None,
    context_labels: dict[str, str] | None,
) -> None:
    target = action.get("target")
    target_label = _object_label(target, object_labels, context_labels)
    if not target_label:
        return
    memory.add_spatial(
        name=target_label,
        node_type="object",
        relations=[{"target": session_name, "type": "acted_on_in"}],
        properties={"from_action_target": True},
    )


def _record_placement_relation(memory, action_text: str, success: bool) -> None:
    if not success:
        return
    match = PLACEMENT_RE.search(action_text or "")
    if not match:
        return
    obj = match.group("object")
    receptacle = match.group("receptacle")
    relation = "in" if re.search(r"\b(?:in|into)\b", match.group(0), re.I) else "on"
    memory.update_spatial(
        name=obj,
        node_type="object",
        relations=[{"target": receptacle, "type": relation}],
        properties={"last_successful_placement": True},
    )


def _visible_label_by_type(
    step: dict,
    obj_type: str,
    object_labels: dict[str, str] | None,
    context_labels: dict[str, str] | None,
) -> str:
    matches: list[tuple[float, str]] = []
    for obj in step.get("visible_objects", []):
        object_id = obj.get("object_id") or obj.get("objectId")
        if object_type(object_id) != obj_type and obj.get("object_type") != obj_type:
            continue
        label = _object_label(object_id, object_labels, context_labels)
        if label:
            matches.append((float(obj.get("distance") or 999.0), label))
    if not matches:
        return obj_type
    return sorted(matches, key=lambda item: (item[0], item[1]))[0][1]


def _action_target_counts(contexts: list[dict]) -> dict[str, int]:
    """Count objects directly manipulated in context actions."""
    counts: dict[str, int] = {}
    direct_action_types = {
        "PickUp",
        "PickupObject",
        "Open",
        "Close",
        "ToggleOn",
        "ToggleOff",
        "Slice",
        "Break",
        "Dirty",
        "Clean",
        "Fill",
        "Empty",
    }
    for session in contexts:
        for step in session.get("steps") or []:
            action = step.get("action") or {}
            action_type = action.get("action_type")
            target = action.get("target")
            if (
                isinstance(action_type, str)
                and action_type in direct_action_types
                and isinstance(target, str)
                and target
            ):
                counts[target] = counts.get(target, 0) + 1
    return counts


def _visible_label_by_type_for_memory_cue(
    step: dict,
    obj_type: str,
    object_labels: dict[str, str] | None,
    context_labels: dict[str, str] | None,
    action_target_counts: dict[str, int] | None,
) -> str:
    candidates: list[tuple[int, float, str]] = []
    for obj in step.get("visible_objects", []):
        object_id = obj.get("object_id") or obj.get("objectId")
        if object_type(object_id) != obj_type and obj.get("object_type") != obj_type:
            continue
        label = _object_label(object_id, object_labels, context_labels)
        if label:
            candidates.append(
                (
                    int((action_target_counts or {}).get(object_id, 0)),
                    float(obj.get("distance") or 999.0),
                    label,
                )
            )
    if not candidates:
        return obj_type
    return sorted(candidates, key=lambda item: (item[0], item[1], item[2]))[0][2]


def _position_distance(a: Any, b: Any) -> float | None:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    try:
        ax, ay, az = float(a.get("x")), float(a.get("y")), float(a.get("z"))
        bx, by, bz = float(b.get("x")), float(b.get("y")), float(b.get("z"))
    except (TypeError, ValueError):
        return None
    return ((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2) ** 0.5


def _visible_receptacle_label_for_cue(
    step: dict,
    obj_type: str,
    receptacle_type: str,
    object_labels: dict[str, str] | None,
    context_labels: dict[str, str] | None,
    selected_object_label: str | None = None,
) -> str:
    object_candidates = [
        obj
        for obj in step.get("visible_objects", [])
        if object_type(obj.get("object_id") or obj.get("objectId")) == obj_type
        or obj.get("object_type") == obj_type
        or obj.get("objectType") == obj_type
    ]
    object_candidates = sorted(
        object_candidates,
        key=lambda obj: (
            0
            if selected_object_label
            and _object_label(
                obj.get("object_id") or obj.get("objectId"),
                object_labels,
                context_labels,
            )
            == selected_object_label
            else 1,
            float(obj.get("distance") or 999.0),
        ),
    )
    if not object_candidates:
        return _visible_label_by_type(
            step, receptacle_type, object_labels, context_labels
        )
    object_position = object_candidates[0].get("position")
    matches: list[tuple[int, float, float, str]] = []
    for obj in step.get("visible_objects", []):
        object_id = obj.get("object_id") or obj.get("objectId")
        visible_type = obj.get("object_type") or obj.get("objectType")
        id_base_matches = object_type(object_id) == receptacle_type
        semantic_matches = visible_type == receptacle_type
        child_receptacle_type = BASE_RECEPTACLE_CHILD_TYPES.get(receptacle_type)
        child_receptacle_matches = bool(
            child_receptacle_type
            and (
                visible_type == child_receptacle_type
                or object_type(object_id) == child_receptacle_type
            )
        )
        if not semantic_matches and not (
            child_receptacle_matches
            or (receptacle_type in CHILD_SLOT_CUE_BASE_TYPES and id_base_matches)
        ):
            continue
        position_distance = _position_distance(object_position, obj.get("position"))
        if position_distance is None:
            continue
        label = _object_label(object_id, object_labels, context_labels)
        if label:
            is_child_slot = bool(
                isinstance(object_id, str)
                and "___" in object_id
                and visible_type
                and visible_type != receptacle_type
                and receptacle_type in CHILD_SLOT_CUE_BASE_TYPES
            )
            # ProcTHOR often represents a semantic receptacle such as a
            # Cabinet/Shelf as a slot on a larger physical ShelvingUnit.  If a
            # passive cue says the object was on a ShelvingUnit, bind the
            # memory to the visible child slot closest to the object instead
            # of the coarse base object; otherwise retrieval can navigate to a
            # sibling shelf and never open the actual cabinet containing it.
            # Some furniture types such as Dresser are also valid top-level
            # support surfaces; for those, a nearby drawer/cabinet slot is not
            # enough evidence to rewrite the passive cue away from the base.
            priority = 0 if (is_child_slot or child_receptacle_matches) else 1
            matches.append(
                (
                    priority,
                    position_distance,
                    float(obj.get("distance") or 999.0),
                    label,
                )
            )
    if not matches:
        return _visible_label_by_type(
            step, receptacle_type, object_labels, context_labels
        )
    return sorted(matches, key=lambda item: (item[0], item[1], item[2], item[3]))[0][3]


def _record_memory_cue(
    memory,
    session_name: str,
    step: dict,
    object_labels: dict[str, str] | None,
    context_labels: dict[str, str] | None,
    action_target_counts: dict[str, int] | None,
    seen_cues: set[str],
    cue_anchors: dict[tuple[str, str, str], tuple[str, str]],
    metadata: dict[str, Any] | None = None,
    label_items: list[tuple[str, str]] | None = None,
) -> None:
    cue = step.get("memory_cue_exposed")
    if not isinstance(cue, dict):
        return
    description = str(cue.get("description") or "").strip()
    if not description:
        return

    description = _replace_ids_with_labels(
        description, object_labels, context_labels, label_items
    )
    cue_text = f"Observed memory cue in {session_name}: {description}"

    match = CUE_LOCATION_RE.search(description)
    if match:
        obj_type = object_type(match.group("object")) or match.group("object")
        receptacle_type = object_type(match.group("receptacle")) or match.group(
            "receptacle"
        )
        anchor_key = (session_name, obj_type.lower(), receptacle_type.lower())
        if anchor_key in cue_anchors:
            obj_label, _ = cue_anchors[anchor_key]
            receptacle_label = _visible_receptacle_label_for_cue(
                step,
                obj_type,
                receptacle_type,
                object_labels,
                context_labels,
                selected_object_label=obj_label,
            )
            cue_anchors[anchor_key] = (obj_label, receptacle_label)
        else:
            obj_label = _visible_label_by_type_for_memory_cue(
                step,
                obj_type,
                object_labels,
                context_labels,
                action_target_counts,
            )
            receptacle_label = _visible_receptacle_label_for_cue(
                step,
                obj_type,
                receptacle_type,
                object_labels,
                context_labels,
                selected_object_label=obj_label,
            )
            cue_anchors[anchor_key] = (obj_label, receptacle_label)
        cue_text = (
            f"Observed memory cue in {session_name}: {obj_label} was seen on/in "
            f"{receptacle_label}. Original note: {description}"
        )
        memory.update_spatial(
            name=obj_label,
            node_type="object",
            relations=[{"target": receptacle_label, "type": "on_or_in"}],
            properties={"memory_cue_observation": True},
        )

    if cue_text in seen_cues:
        return
    seen_cues.add(cue_text)
    event_metadata = dict(metadata or {})
    event_metadata.update(
        {
            "raw_action_text": cue_text,
            "raw_feedback_text": description,
            "memory_phase": "context_memory_cue",
        }
    )
    memory.add_event(
        action=cue_text,
        success=True,
        feedback=description,
        event_summary=cue_text,
        metadata=event_metadata,
    )


def _simple_family_key(value: Any) -> str:
    value = strip_object_ids(str(value or "")).strip()
    if not value:
        return ""
    family = object_type(value)
    family = re.sub(r"_\d+$", "", str(family or ""))
    return family.lower()


def _resolve_relocation_cue_object_and_target(
    memory,
    *,
    namespace: str | None,
    obj_type: str,
    source: str,
    target: str,
) -> tuple[str, str]:
    obj_key = _simple_family_key(obj_type)
    source_key = _simple_family_key(source)
    target_key = _simple_family_key(target)
    experience = getattr(memory, "experience", None)
    latest_locations = getattr(experience, "latest_locations", {}) if experience else {}
    candidates: list[tuple[int, int, int, int, str, str]] = []
    for item in latest_locations.values():
        if namespace and item.get("namespace") != namespace:
            continue
        if (
            item.get("object_family_key") != obj_key
            and item.get("object_key") != obj_key
        ):
            continue
        for history in item.get("location_history") or []:
            history_target = str(history.get("target") or "")
            target_match = int(
                bool(target_key and _simple_family_key(history_target) == target_key)
            )
            previous_key = _simple_family_key(history.get("previous_target"))
            source_match = int(bool(source_key and previous_key == source_key))
            if not (target_match or source_match):
                continue
            object_name = str(item.get("object") or obj_type)
            candidates.append(
                (
                    source_match,
                    target_match,
                    int(_target_is_instance_specific(object_name)),
                    int(history.get("step") or -1),
                    object_name,
                    history_target or target,
                )
            )
    if not candidates:
        return obj_type, target
    candidates.sort(key=lambda item: (-item[1], -item[0], -item[2], -item[3], item[4]))
    return candidates[0][4], candidates[0][5]


def _record_description_relocation_cues(
    memory,
    *,
    description: str,
    metadata: dict[str, Any],
) -> None:
    if not description:
        return
    namespace = metadata.get("memory_namespace")
    for match in CUE_RELOCATION_RE.finditer(description):
        obj = match.group("object")
        source = match.group("source")
        target = match.group("target")
        resolved_obj, resolved_target = _resolve_relocation_cue_object_and_target(
            memory,
            namespace=namespace,
            obj_type=obj,
            source=source,
            target=target,
        )
        experience = getattr(memory, "experience", None)
        if not experience:
            continue
        experience.add_observed_location(
            obj=resolved_obj,
            target=resolved_target,
            relation="on_or_in",
            namespace=namespace,
            metadata={**metadata, "memory_phase": "context_description_cue"},
            event_ids=[f"description_cue_{memory.current_step}"],
            step=memory.current_step,
            source="explicit_memory_cue",
            previous_target=source,
        )


def _preload_contexts(
    memory,
    contexts: list[dict],
    object_labels: dict[str, str] | None = None,
) -> None:
    context_labels = _context_object_labels(contexts)
    label_items = _merged_label_items(object_labels, context_labels)
    action_target_counts = _action_target_counts(contexts)
    for session_idx, session in enumerate(contexts):
        raw_name = session.get("session_name") or f"context_session_{session_idx}"
        session_name = strip_object_ids(str(raw_name))
        metadata = {
            "memory_namespace": session.get("memory_namespace"),
            "source_episode_id": session.get("memory_namespace"),
            "session_name": session_name,
            "scene": session.get("scene"),
            "room_type": session.get("room_type"),
        }
        description = sanitize_context_description(session.get("description") or "")
        steps = list(session.get("steps") or [])

        memory.add_scene(
            caption=f"{session_name}: {description}",
            visible_objects=_visible_object_names(
                steps[0], object_labels, context_labels
            )
            if steps
            else [],
        )
        memory.add_spatial(
            name=session_name,
            node_type="room",
            properties={"context_session": True},
        )

        seen_cues: set[str] = set()
        cue_anchors: dict[tuple[str, str, str], tuple[str, str]] = {}
        for step in steps:
            action = step.get("action") or {}
            action_text, success, feedback_text = _action_text(
                step,
                object_labels,
                context_labels,
                label_items,
            )
            feedback = step.get("feedback") or {}
            event_metadata = dict(metadata)
            event_metadata.update(
                {
                    "raw_action_target": action.get("target"),
                    "raw_action_target_type": action.get("action_type"),
                    "raw_action_text": _replace_ids_with_labels(
                        action.get("natural_language")
                        or action.get("action_type")
                        or "Unknown action",
                        object_labels,
                        context_labels,
                        label_items,
                    ),
                    "raw_feedback_text": _replace_ids_with_labels(
                        feedback.get("message") or "",
                        object_labels,
                        context_labels,
                        label_items,
                    ),
                }
            )
            event_summary = (
                f"Session {session_name}; action: {action_text}; "
                f"result: {'success' if success else 'failed'}"
            )
            if feedback_text:
                event_summary += f"; feedback: {feedback_text}"
            memory.add_event(
                action=action_text,
                success=success,
                feedback=feedback_text,
                event_summary=event_summary,
                metadata=event_metadata,
            )
            if step.get("step_id") in {0, 1} or action.get("action_type") in {
                "Navigate",
                "PickUp",
                "PutObject",
                "Open",
                "Close",
                "HumanIntervention",
            }:
                _record_visible_objects(
                    memory, session_name, step, object_labels, context_labels
                )
            _record_memory_cue(
                memory,
                session_name,
                step,
                object_labels,
                context_labels,
                action_target_counts,
                seen_cues,
                cue_anchors,
                metadata=event_metadata,
                label_items=label_items,
            )
            _record_target_object(
                memory, action, session_name, object_labels, context_labels
            )
            _record_placement_relation(memory, action_text, success)
            memory.step()
        _record_description_relocation_cues(
            memory,
            description=description,
            metadata=metadata,
        )


def _disable_embeddings(memory) -> None:
    memory.spatial._emb = None
    memory.scene._emb = None
    memory.event._emb = None


def _safe_build_memory_text(
    build_memory_context, memory, config: dict, instruction: str
) -> str:
    try:
        return build_memory_context(memory, config, memory.current_step, instruction)
    except (ImportError, ModuleNotFoundError):
        _disable_embeddings(memory)
        return build_memory_context(memory, config, memory.current_step, instruction)


def _normalize_online_action_text(text: Any) -> str:
    action = strip_object_ids(str(text or ""))
    action = action.replace("on/in", "on")
    action = action.replace("Put held object", "Put Object")
    return action


def _online_action_signature(text: Any) -> tuple[str, str] | None:
    action = strip_object_ids(str(text or "")).strip()
    patterns = [
        ("PutObject", r"^Put held object on/in (?P<target>.+)$"),
        ("PutObject", r"^Put Object on (?P<target>.+)$"),
        ("Navigate", r"^Navigate to (?P<target>.+)$"),
        ("PickUp", r"^Pick up (?P<target>.+)$"),
        ("Open", r"^Open (?P<target>.+)$"),
        ("Close", r"^Close (?P<target>.+)$"),
    ]
    for action_type, pattern in patterns:
        match = re.match(pattern, action)
        if match:
            target = match.group("target").strip()
            if target:
                return action_type, target
    return None


def _target_family(value: Any) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    family = strip_object_ids(object_type(value))
    family = re.sub(r"_\d+$", "", str(family or ""))
    return family.lower()


def _target_exact(value: Any) -> str:
    return str(value or "").strip().lower()


def _target_is_instance_specific(value: Any) -> bool:
    text = str(value or "")
    return bool("|" in text or re.search(r"_\d+(?:$|[^A-Za-z0-9])", text))


def _target_physical_base(value: Any) -> str:
    text = _target_exact(value)
    if not text:
        return ""
    return text.split("___", 1)[0]


def _target_matches_action(target: Any, action: dict) -> bool:
    target_exact = _target_exact(target)
    if not target_exact:
        return False
    exact_values = {
        _target_exact(action.get("target_label")),
        _target_exact(action.get("target_type")),
    }
    if _target_is_instance_specific(target):
        return target_exact in exact_values
    family_values = {
        _target_family(action.get("target_label")),
        _target_family(action.get("target_type")),
    }
    return target_exact in exact_values or target_exact in family_values


def _target_matches_action_target_id(target: Any, action: dict) -> bool:
    target_exact = _target_exact(target)
    if not target_exact:
        return False
    action_target = action.get("target")
    if _target_is_instance_specific(target):
        return target_exact == _target_exact(action_target)
    return target_exact == _target_exact(
        action_target
    ) or target_exact == _target_family(action_target)


def _target_matches_action_child_receptacle(target: Any, action: dict) -> bool:
    if _target_is_instance_specific(target):
        return False
    aliases = BASE_RECEPTACLE_CHILD_FAMILIES.get(_target_family(target), set())
    if not aliases:
        return False
    action_families = {
        _target_family(action.get("target")),
        _target_family(action.get("target_label")),
        _target_family(action.get("target_type")),
    }
    return bool(aliases.intersection(action_families))


def _target_matches_action_parent(target: Any, action: dict) -> bool:
    target_exact = _target_exact(target)
    if not target_exact:
        return False
    exact_values = {
        _target_exact(action.get("parent_target")),
        _target_exact(action.get("parent_target_label")),
        _target_exact(action.get("parent_target_type")),
    }
    if _target_is_instance_specific(target):
        return target_exact in exact_values
    family_values = {
        _target_family(action.get("parent_target")),
        _target_family(action.get("parent_target_label")),
        _target_family(action.get("parent_target_type")),
    }
    return target_exact in exact_values or target_exact in family_values


def _target_matches_action_parent_child_receptacle(target: Any, action: dict) -> bool:
    if _target_is_instance_specific(target):
        return False
    aliases = BASE_RECEPTACLE_CHILD_FAMILIES.get(_target_family(target), set())
    if not aliases:
        return False
    parent_families = {
        _target_family(action.get("parent_target")),
        _target_family(action.get("parent_target_label")),
        _target_family(action.get("parent_target_type")),
    }
    return bool(aliases.intersection(parent_families))


def _action_is_blocked(
    action: dict,
    blocked_signatures: set[tuple[str, str]],
    *,
    match_target_id: bool = True,
) -> bool:
    action_type = str(action.get("action_type") or "")
    for blocked_type, blocked_target in blocked_signatures:
        if str(blocked_type or "") != action_type:
            continue
        if _target_matches_action(blocked_target, action):
            return True
        if match_target_id and _target_matches_action_target_id(blocked_target, action):
            return True
    return False


def _storage_recovery_block_types() -> set[str]:
    return {"Navigate", "Open", "Close", "PutObject", "TransferContents"}


def _agent_is_holding(obs: dict) -> bool:
    state = obs.get("agent_state") or {}
    if not isinstance(state, dict):
        return False
    holding = (
        state.get("holding") or state.get("held_object") or state.get("heldObject")
    )
    return bool(holding)


def _instruction_mentions_target_family(instruction: str, family: str) -> bool:
    text = str(instruction or "").lower()
    family = str(family or "").lower()
    if not text or not family:
        return False
    if re.search(rf"(?<![a-z0-9]){re.escape(family)}(?![a-z0-9])", text):
        return True
    # Multi-token object names may appear with spaces or punctuation
    # (for example "remote control" vs "RemoteControl").  Keep this fallback
    # for longer families, but avoid short substrings such as "bed" in
    # "Bedroom".
    if len(family) >= 5:
        normalized_instruction = re.sub(r"[^a-z0-9]+", "", text)
        return family in normalized_instruction
    return False


def _instruction_pickup_target_families(
    instruction: str, actions: list[dict]
) -> set[str]:
    if not str(instruction or "").strip():
        return set()

    families = {
        _target_family(
            action.get("target_type")
            or action.get("target_label")
            or action.get("target")
        )
        for action in actions
        if action.get("action_type") in {"Navigate", "PickUp"}
    }
    return {
        family
        for family in families
        if family and _instruction_mentions_target_family(instruction, family)
    }


def _instruction_is_retrieval_goal(instruction: str) -> bool:
    text = str(instruction or "").lower()
    if not text or _instruction_is_placement_goal(text):
        return False
    if re.search(r"\b(?:bring|get|retrieve|fetch|take)\b", text):
        return True
    return any(
        term in text for term in ("拿过来", "拿来", "取回", "取来", "带过来", "带来")
    )


def _instruction_retrieval_target_families(
    instruction: str,
    actions: list[dict],
) -> set[str]:
    if not _instruction_is_retrieval_goal(instruction):
        return set()
    if not str(instruction or "").strip():
        return set()
    families = {
        _target_family(
            action.get("target_type")
            or action.get("target_label")
            or action.get("target")
        )
        for action in actions
        if action.get("action_type") not in {None, "Done", "Stop"}
    }
    return {
        family
        for family in families
        if family and _instruction_mentions_target_family(instruction, family)
    }


def _instruction_is_placement_goal(instruction: str) -> bool:
    text = str(instruction or "").lower()
    if not text:
        return False
    if re.search(r"\b(?:put|place|store|stash|hide|move)\b", text):
        return True
    return any(term in text for term in ("放", "藏", "存放", "收纳", "放进", "放入"))


FAILURE_STORAGE_TARGET_FAMILIES = {
    "cabinet",
    "drawer",
    "dresser",
    "fridge",
    "microwave",
    "safe",
}


def _instruction_requires_openable_storage(instruction: str) -> bool:
    text = str(instruction or "").lower()
    if not text:
        return False
    if any(term in text for term in ("收纳容器", "能打开", "隐蔽", "藏起来")):
        return True
    return bool(
        re.search(r"\b(?:hide|stash)\b", text)
        and re.search(r"\b(?:container|storage|openable)\b", text)
    )


def _action_matches_failure_storage_target(action: dict) -> bool:
    return any(
        _target_family(value) in FAILURE_STORAGE_TARGET_FAMILIES
        for value in (
            action.get("target_label"),
            action.get("target_type"),
            action.get("target"),
        )
    )


def _holding_placement_progress_available(actions: list[dict]) -> bool:
    return any(
        action.get("action_type") in {"Navigate", "Open", "PutObject"}
        for action in actions
    )


def _has_preferred_memory_action(
    actions: list[dict],
    *,
    preferred_put_targets: set[str],
    preferred_physical_put_targets: set[str],
) -> bool:
    if not preferred_put_targets and not preferred_physical_put_targets:
        return False
    for action in actions:
        action_type = action.get("action_type")
        if action_type == "PutObject":
            if any(
                _target_matches_action_target_id(target, action)
                for target in preferred_physical_put_targets
            ):
                return True
            if any(
                _target_matches_action(target, action)
                for target in preferred_put_targets
            ):
                return True
        if action_type in {"Navigate", "Open"} and any(
            _target_matches_action_target_id(target, action)
            for target in preferred_physical_put_targets
        ):
            return True
        if action_type in {"Navigate", "Open"} and any(
            _target_matches_action(target, action) for target in preferred_put_targets
        ):
            return True
    return False


def _habit_policy_targets(habit: dict) -> tuple[set[str], set[str], set[str], bool]:
    policy = habit.get("placement_policy") or {}
    physical = {
        str(value)
        for value in (policy.get("preferred_physical_target_families") or [])
        if value
    }
    preferred = {
        str(value)
        for value in (
            policy.get("semantic_target_families")
            or policy.get("preferred_target_families")
            or [policy.get("preferred_target_key")]
        )
        if value
    }
    blocked = {
        str(value)
        for value in (
            policy.get("blocked_target_families")
            or policy.get("blocked_target_keys")
            or habit.get("negative_targets")
            or []
        )
        if value
    }
    object_target_support = int(habit.get("object_target_support_count") or 0)
    exact_matches = set(habit.get("matched_exact_object_terms") or [])
    strict_physical = bool(physical and object_target_support > 0 and exact_matches)
    if physical and not strict_physical:
        semantic_only = preferred - physical
        if semantic_only:
            preferred = semantic_only
    return preferred, physical, blocked, strict_physical


def _context_from_memory_text(memory_text: str, stats: dict[str, Any]) -> list[dict]:
    return [
        {
            "session_name": "embodied_memorizer retrieved context",
            "scene": "embodied_memorizer",
            "room_type": None,
            "description": memory_text,
            "total_steps": 0,
            "steps": [],
            "memory_module": stats,
        }
    ]


def _instruction_query_terms(instruction: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    stop = {
        "please",
        "store",
        "this",
        "location",
        "matches",
        "household",
        "implicit",
        "habit",
        "best",
        "the",
        "that",
        "where",
        "object",
        "current",
        "task",
    }
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9_]*", instruction or ""):
        pieces = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", raw)
        candidates = [raw]
        if len(pieces) > 1:
            candidates.append(" ".join(pieces))
            candidates.append(pieces[0])
        for term in candidates:
            lowered = term.lower()
            if len(lowered) < 3 or lowered in stop or lowered in seen:
                continue
            seen.add(lowered)
            terms.append(term)
    return terms[:8]


def _memory_result_text(item: dict) -> str:
    return " ".join(
        str(item.get(key) or "")
        for key in ("action", "feedback", "event_summary", "name", "caption")
    ).lower()


def _matches_query_text(item: dict, query: str) -> bool:
    text = _memory_result_text(item)
    lowered = query.lower()
    if lowered in text:
        return True
    terms = [t for t in re.findall(r"[a-zA-Z]+", lowered) if len(t) >= 4]
    return any(term in text for term in terms)


def _collect_memory_results(
    query_fn,
    queries: list[str],
    total_k: int,
    per_query_k: int,
    *,
    exact_queries: set[str] | None = None,
) -> list[dict]:
    results: list[dict] = []
    seen: set[str] = set()
    exact_queries = exact_queries or set()
    for query in queries:
        if len(results) >= total_k:
            break
        try:
            matches = query_fn(query, per_query_k)
        except (ImportError, ModuleNotFoundError):
            raise
        for item in matches:
            if query in exact_queries and not _matches_query_text(item, query):
                continue
            key = str(item.get("id") or item)
            if key in seen:
                continue
            seen.add(key)
            results.append(item)
            if len(results) >= total_k:
                break
    return results


def _safe_collect_memory_results(
    memory, layer: str, queries: list[str], total_k: int
) -> list[dict]:
    query_fn = {
        "event": memory.query_event,
        "spatial": memory.query_spatial,
        "scene": memory.query_scene,
    }[layer]
    exact_queries = set(queries) if layer == "event" else set()
    try:
        return _collect_memory_results(
            query_fn,
            queries,
            total_k=total_k,
            per_query_k=4,
            exact_queries=exact_queries,
        )
    except (ImportError, ModuleNotFoundError):
        _disable_embeddings(memory)
        return _collect_memory_results(
            query_fn,
            queries,
            total_k=total_k,
            per_query_k=4,
            exact_queries=exact_queries,
        )


def _latest_placement_facts(memory, terms: list[str], limit: int = 8) -> list[str]:
    """Surface embodied_memorizer spatial nodes marked as latest successful placements."""
    spatial = getattr(memory, "spatial", None)
    nodes = getattr(spatial, "nodes", {}) or {}
    edges = getattr(spatial, "edges", {}) or {}
    if not isinstance(nodes, dict) or not isinstance(edges, dict):
        return []

    term_set = {str(term).lower() for term in terms if term}
    facts: list[tuple[int, str]] = []
    seen: set[str] = set()
    for node_id, node in nodes.items():
        if not isinstance(node, dict):
            continue
        props = node.get("properties") or {}
        if not props.get("last_successful_placement"):
            continue
        source = node.get("name")
        if not source:
            continue
        for (src_id, dst_id), edge in edges.items():
            if src_id != node_id or not isinstance(edge, dict):
                continue
            relation = edge.get("relation")
            if relation not in {"on", "in", "on_or_in"}:
                continue
            target_node = nodes.get(dst_id) or {}
            target = target_node.get("name")
            if not target:
                continue
            fact = f"{source} is currently {relation} {target} after the latest successful placement."
            if fact in seen:
                continue
            seen.add(fact)
            priority = 0 if str(source).lower() in term_set else 1
            facts.append((priority, fact))
    facts.sort(key=lambda item: item[0])
    return [fact for _, fact in facts[:limit]]


def _build_retrieved_memory_text(
    memory,
    format_query_results,
    instruction: str,
    contexts: list[dict],
    current_scene: Any = None,
    current_namespace: str | None = None,
    *,
    experience_top_k: int = 5,
    event_top_k: int,
    spatial_top_k: int,
    scene_top_k: int,
) -> str:
    terms = _instruction_query_terms(instruction)
    query_relevant = getattr(memory, "query_relevant", None)
    if callable(query_relevant):
        try:
            results = query_relevant(
                instruction,
                namespace=current_namespace,
                experience_top_k=experience_top_k,
                event_top_k=8,
                spatial_top_k=spatial_top_k,
                scene_top_k=scene_top_k,
            )
        except (ImportError, ModuleNotFoundError):
            _disable_embeddings(memory)
            results = query_relevant(
                instruction,
                namespace=current_namespace,
                experience_top_k=experience_top_k,
                event_top_k=8,
                spatial_top_k=spatial_top_k,
                scene_top_k=scene_top_k,
            )
        if results:
            return "## embodied_memorizer Retrieved Memory\n" + format_query_results(
                results
            )

    event_queries = [
        "preference violation",
        "human intervention",
        "failed placement",
        "failed action",
        "Try placing",
        "Place",
        instruction,
        *terms,
    ]
    spatial_queries = [instruction, *terms, "last successful placement"]
    scene_queries = [instruction, *terms[:3]]

    event_results = _safe_collect_memory_results(
        memory, "event", event_queries, total_k=event_top_k
    )
    has_correction_evidence = any(
        "preference violation" in _memory_result_text(item)
        or "human intervention" in _memory_result_text(item)
        for item in event_results
    )
    if has_correction_evidence:
        event_results = [
            item
            for item in event_results
            if (
                "preference violation" in _memory_result_text(item)
                or "human intervention" in _memory_result_text(item)
                or str(item.get("action") or "").lower().startswith("try placing")
                or str(item.get("action") or "").lower().startswith("place")
            )
        ]
    spatial_results = []
    if not has_correction_evidence:
        spatial_results = _safe_collect_memory_results(
            memory, "spatial", spatial_queries, total_k=spatial_top_k
        )
    scene_results = _safe_collect_memory_results(
        memory, "scene", scene_queries, total_k=scene_top_k
    )

    sections: list[str] = []
    latest_facts = _latest_placement_facts(memory, terms, limit=8)
    if latest_facts:
        sections.append(
            "## embodied_memorizer Latest Placement Facts\n"
            "Treat these as the highest-priority current object-location facts:\n"
            + "\n".join(f"- {fact}" for fact in latest_facts)
        )
    if event_results:
        sections.append(
            "## embodied_memorizer Retrieved Events\n"
            + format_query_results(event_results)
        )
    if spatial_results:
        sections.append(
            "## embodied_memorizer Retrieved Spatial Memory\n"
            + format_query_results(spatial_results)
        )
    if scene_results:
        sections.append(
            "## embodied_memorizer Retrieved Scene Memory\n"
            + format_query_results(scene_results)
        )
    if not sections:
        return ""
    return "\n\n".join(sections)


def _location_policy_target_sets(
    locations: list[dict],
) -> tuple[set[str], set[str], set[str]]:
    object_targets: set[str] = set()
    exact_object_targets: set[str] = set()
    receptacle_targets: set[str] = set()
    matched_locations = [
        item
        for item in locations
        if item.get("experience_type") == "observed_object_location"
        and item.get("matched_object_terms")
    ]
    for item in matched_locations[:1]:
        policy = item.get("location_policy") or {}
        for target in policy.get("preferred_exact_object_targets") or [
            policy.get("preferred_object_key"),
            item.get("object_key"),
        ]:
            if target:
                exact_object_targets.add(str(target))
        for target in (
            policy.get("fallback_object_families")
            or policy.get("preferred_object_families")
            or [
                policy.get("preferred_object_key"),
                policy.get("preferred_object_family_key"),
                item.get("object_key"),
                item.get("object_family_key"),
            ]
        ):
            if target:
                object_targets.add(str(target))
        for target in policy.get("preferred_location_targets") or [
            policy.get("preferred_location_target_key"),
            policy.get("preferred_location_family_key"),
            item.get("target_key"),
            item.get("target_family_key"),
        ]:
            if target:
                receptacle_targets.add(str(target))
    return object_targets, exact_object_targets, receptacle_targets


def _prefer_current_scene_location(
    locations: list[dict],
    current_scene: Any,
) -> list[dict]:
    """Prefer the same queried object in the active action-space scene."""
    scene = str(current_scene or "")
    if not scene or not locations:
        return locations
    leading_object_key = str(locations[0].get("object_key") or "")
    preferred = [
        item
        for item in locations
        if str(item.get("last_scene") or "") == scene
        and (
            not leading_object_key
            or str(item.get("object_key") or "") == leading_object_key
        )
    ]
    if not preferred:
        return locations
    preferred_ids = {id(item) for item in preferred}
    return preferred + [item for item in locations if id(item) not in preferred_ids]


class MemoryRuntime:
    """Episode-local embodied_memorizer state with online write/retrieve updates."""

    def __init__(
        self,
        contexts: list[dict],
        instruction: str,
        *,
        object_labels: dict[str, str] | None = None,
        current_scene: Any = None,
        current_namespace: str | None = None,
        memory_module_root: str | None = None,
        embedding_model: str | None = None,
        embedding_device: str = "cpu",
        max_memory_tokens: int = 1600,
        event_top_k: int = 14,
        spatial_top_k: int = 12,
        scene_top_k: int = 3,
        preload_contexts: bool = True,
        enable_action_filtering: bool = True,
        enable_online_writeback: bool = True,
        enable_progress_action_guard: bool = True,
        enable_spatial_identity_scope: bool = True,
        enable_relation_aware_spatial: bool = True,
        enable_consolidated_interaction_retrieval: bool = True,
        prefer_current_scene_location_records: bool = False,
    ):
        (
            MemorySystem,
            DEFAULT_CONFIG,
            merge_config,
            build_memory_context,
            format_query_results,
        ) = _ensure_memory_module(memory_module_root)
        if embedding_model is None and Path(DEFAULT_LOCAL_EMBEDDING_MODEL).exists():
            embedding_model = DEFAULT_LOCAL_EMBEDDING_MODEL
        self.config = _memory_config(
            default_config=DEFAULT_CONFIG,
            merge_config=merge_config,
            embedding_model=embedding_model,
            embedding_device=embedding_device,
            max_memory_tokens=max_memory_tokens,
            event_top_k=event_top_k,
            spatial_top_k=spatial_top_k,
            scene_top_k=scene_top_k,
        )
        self.memory = MemorySystem(self.config)
        self.memory.spatial.relation_aware_updates = enable_relation_aware_spatial
        self.memory.spatial.directional_incoming_relations = (
            enable_relation_aware_spatial
        )
        self.memory.experience.consolidated_portfolio_only = (
            enable_consolidated_interaction_retrieval
        )
        if preload_contexts:
            _preload_contexts(self.memory, contexts, object_labels=object_labels)
        self.instruction = instruction
        self.contexts = contexts
        self.preload_contexts = preload_contexts
        self.current_scene = current_scene
        self.current_namespace = current_namespace
        self.embedding_model = embedding_model or DEFAULT_CONFIG.get("embedding_model")
        self.embedding_device = embedding_device
        self.memory_module_root = str(
            Path(
                memory_module_root
                or os.getenv("MEMORY_MODULE_ROOT")
                or DEFAULT_MEMORY_MODULE_ROOT
            )
        )
        self.build_memory_context = build_memory_context
        self.format_query_results = format_query_results
        self.event_top_k = event_top_k
        self.spatial_top_k = spatial_top_k
        self.scene_top_k = scene_top_k
        self.enable_action_filtering = enable_action_filtering
        self.enable_online_writeback = enable_online_writeback
        self.enable_progress_action_guard = enable_progress_action_guard
        self.enable_spatial_identity_scope = enable_spatial_identity_scope
        self.enable_relation_aware_spatial = enable_relation_aware_spatial
        self.enable_consolidated_interaction_retrieval = (
            enable_consolidated_interaction_retrieval
        )
        self.prefer_current_scene_location_records = (
            prefer_current_scene_location_records
        )
        infer_intent = getattr(
            getattr(self.memory, "experience", None), "infer_query_intent", None
        )
        self.memory_query_intent = (
            infer_intent(instruction) if callable(infer_intent) else "generic"
        )
        self.online_step_count = 0
        self.blocked_action_signatures: set[tuple[str, str]] = set()
        self.memory_blocked_action_signatures: set[tuple[str, str]] = set()
        self.memory_habit_blocked_action_signatures: set[tuple[str, str]] = set()
        self.memory_preferred_put_targets: set[str] = set()
        self.memory_preferred_physical_put_targets: set[str] = set()
        self.memory_deprioritized_physical_put_targets: set[str] = set()
        self.memory_location_object_targets: set[str] = set()
        self.memory_location_exact_object_targets: set[str] = set()
        self.memory_location_receptacle_targets: set[str] = set()
        self._last_task_progress = 0.0
        self._no_progress_action_counts: dict[tuple[str, str], int] = {}
        self._refresh_memory_blocked_action_signatures()

    def observe_step(self, info: dict) -> None:
        action_text = _normalize_online_action_text(info.get("action_description"))
        feedback_text = strip_object_ids(str(info.get("env_feedback") or ""))
        if not action_text:
            return
        action_entry = info.get("action_entry") or info.get("action") or {}
        if not isinstance(action_entry, dict):
            action_entry = {}
        success = float(info.get("last_action_success") or 0.0) > 0.0
        signature = _online_action_signature(info.get("action_description"))
        if (
            not success
            and signature
            and signature[0] == "PutObject"
            and (
                "no valid positions" in feedback_text.lower()
                or "failed to put object" in feedback_text.lower()
            )
        ):
            _, target = signature
            for action_type in _storage_recovery_block_types():
                self.blocked_action_signatures.add((action_type, target))
        if not success and signature and signature[0] == "Navigate":
            self.blocked_action_signatures.add(signature)
        try:
            current_progress = float(info.get("task_progress") or 0.0)
        except (TypeError, ValueError):
            current_progress = self._last_task_progress
        if (
            self.enable_progress_action_guard
            and signature
            and success
            and current_progress <= self._last_task_progress + 1e-6
        ):
            count = self._no_progress_action_counts.get(signature, 0) + 1
            self._no_progress_action_counts[signature] = count
            if signature[0] in {"Navigate", "Open", "Close"} and count >= 1:
                self.blocked_action_signatures.add(signature)
            if (
                signature[0] == "PutObject"
                and count >= 1
                and _instruction_is_placement_goal(self.instruction)
            ):
                _, target = signature
                for action_type in _storage_recovery_block_types():
                    self.blocked_action_signatures.add((action_type, target))
        elif self.enable_progress_action_guard and signature:
            self._no_progress_action_counts.pop(signature, None)
        self._last_task_progress = max(self._last_task_progress, current_progress)
        if not getattr(self, "enable_online_writeback", True):
            return
        self.memory.add_event(
            action=action_text,
            success=success,
            feedback=feedback_text,
            event_summary=(
                f"Current probe action: {action_text}; "
                f"result: {'success' if success else 'failed'}; "
                f"feedback: {feedback_text}"
            ),
            metadata={
                "memory_namespace": self.current_namespace,
                "source_episode_id": self.current_namespace,
                "session_name": "current_probe_online",
                "scene": self.current_scene,
                "memory_phase": "current_probe",
                "raw_action_text": action_text,
                "raw_action_target": action_entry.get("target"),
                "raw_action_target_type": action_entry.get("action_type"),
                "raw_feedback_text": feedback_text,
            },
        )
        self.memory.step()
        self._refresh_memory_blocked_action_signatures()
        self.online_step_count += 1

    def _refresh_memory_blocked_action_signatures(self) -> None:
        signatures: set[tuple[str, str]] = set()
        habit_signatures: set[tuple[str, str]] = set()
        preferred_put_targets: set[str] = set()
        preferred_physical_put_targets: set[str] = set()
        deprioritized_physical_put_targets: set[str] = set()
        states = []
        try:
            query_states = getattr(self.memory, "query_interaction_states", None)
            if callable(query_states):
                states = query_states(
                    "avoid locked failed invalid interaction target",
                    top_k=50,
                    namespace=self.current_namespace,
                )
        except Exception:
            states = []
        for item in states:
            if item.get("experience_type") != "interaction_state":
                continue
            if item.get("current_status") != "blocked":
                continue
            action_types = {
                str(action_type)
                for action_type in (item.get("blocked_action_types") or [])
                if action_type
            }
            if not action_types:
                action_types = {str(item.get("action_type") or "")}
            if "Open" in action_types:
                action_types.add("Navigate")
            targets = {
                str(item.get("target") or ""),
                *[str(value) for value in (item.get("target_variants") or [])],
            }
            targets = {target for target in targets if target}
            exact_targets = {
                target for target in targets if _target_is_instance_specific(target)
            }
            targets = exact_targets or targets
            for action_type in action_types:
                for target in targets:
                    signatures.add((action_type, target))

        try:
            constraints = self.memory.query_constraints(
                "avoid locked failed invalid interaction target",
                top_k=50,
                namespace=self.current_namespace,
            )
        except Exception:
            constraints = []
        for item in constraints:
            if item.get("experience_type") != "interaction_failure_constraint":
                continue
            if item.get("current_status") == "available" or item.get(
                "superseded_by_success"
            ):
                continue
            action_type = str(item.get("action_type") or "")
            modes = set((item.get("failure_modes") or {}).keys())
            targets = {
                str(item.get("target") or ""),
                *[str(value) for value in (item.get("target_variants") or [])],
            }
            targets = {target for target in targets if target}
            exact_targets = {
                target for target in targets if _target_is_instance_specific(target)
            }
            targets = exact_targets or targets
            if action_type == "Open" and "locked" in modes:
                blocked_types = {"Navigate", "Open", "PutObject", "TransferContents"}
            elif action_type == "PutObject" and "invalid_placement" in modes:
                blocked_types = {"PutObject", "TransferContents"}
            else:
                continue
            for blocked_type in blocked_types:
                for target in targets:
                    signatures.add((blocked_type, target))

        try:
            habits = self.memory.query_habits(
                self.instruction,
                top_k=5,
                namespace=self.current_namespace,
            )
        except Exception:
            habits = []
        for item in habits:
            if item.get("experience_type") != "corrected_placement_habit":
                continue
            (
                preferred_targets,
                physical_targets,
                blocked_targets,
                strict_physical_targets,
            ) = _habit_policy_targets(item)
            for target in preferred_targets:
                preferred_put_targets.add(str(target))
            if strict_physical_targets:
                for target in physical_targets:
                    preferred_physical_put_targets.add(str(target))
            else:
                for target in physical_targets:
                    deprioritized_physical_put_targets.add(str(target))
            for target in blocked_targets:
                habit_signatures.add(("PutObject", str(target)))
        if self.memory_query_intent in {"location", "mixed"}:
            try:
                locations = self.memory.query_locations(
                    self.instruction,
                    top_k=3,
                    namespace=self.current_namespace,
                )
            except Exception:
                locations = []
        else:
            locations = []
        if self.prefer_current_scene_location_records:
            locations = _prefer_current_scene_location(locations, self.current_scene)
        (
            location_object_targets,
            location_exact_object_targets,
            location_receptacle_targets,
        ) = _location_policy_target_sets(locations)
        self.memory_blocked_action_signatures = signatures
        self.memory_habit_blocked_action_signatures = habit_signatures
        self.memory_preferred_put_targets = preferred_put_targets
        self.memory_preferred_physical_put_targets = preferred_physical_put_targets
        self.memory_deprioritized_physical_put_targets = (
            deprioritized_physical_put_targets
        )
        self.memory_location_object_targets = location_object_targets
        self.memory_location_exact_object_targets = location_exact_object_targets
        self.memory_location_receptacle_targets = location_receptacle_targets

    def filter_observation(self, obs: dict) -> dict:
        if not getattr(self, "enable_action_filtering", True):
            return obs
        exact_blocked = (
            self.blocked_action_signatures | self.memory_blocked_action_signatures
        )
        habit_blocked = self.memory_habit_blocked_action_signatures
        blocked = exact_blocked | habit_blocked

        def action_is_blocked(action: dict) -> bool:
            return _action_is_blocked(action, exact_blocked) or _action_is_blocked(
                action,
                habit_blocked,
                match_target_id=False,
            )

        preferred_put_targets = self.memory_preferred_put_targets
        preferred_physical_put_targets = self.memory_preferred_physical_put_targets
        deprioritized_physical_put_targets = (
            self.memory_deprioritized_physical_put_targets
        )
        location_object_targets = self.memory_location_object_targets
        location_exact_object_targets = self.memory_location_exact_object_targets
        location_receptacle_targets = self.memory_location_receptacle_targets
        explicit_task_object = obs.get("task_object_label") or obs.get(
            "task_object_type"
        )
        if explicit_task_object and self.memory_query_intent in {"location", "mixed"}:
            try:
                explicit_locations = self.memory.query_locations(
                    str(explicit_task_object),
                    top_k=3,
                    namespace=self.current_namespace,
                )
            except Exception:
                explicit_locations = []
            explicit_object_key = _target_exact(explicit_task_object)
            exact_locations = [
                item
                for item in explicit_locations
                if _target_exact(item.get("object_key")) == explicit_object_key
            ]
            if self.prefer_current_scene_location_records:
                exact_locations = _prefer_current_scene_location(
                    exact_locations,
                    self.current_scene,
                )
            if exact_locations:
                (
                    location_object_targets,
                    location_exact_object_targets,
                    location_receptacle_targets,
                ) = _location_policy_target_sets(exact_locations[:1])
        actions = obs.get("available_actions", [])
        exact_location_receptacle_targets = {
            target
            for target in location_receptacle_targets
            if _target_is_instance_specific(target)
        }
        location_receptacle_physical_bases: set[str] = set()
        if exact_location_receptacle_targets:
            for action in actions:
                if action.get("action_type") not in {"Navigate", "Open"}:
                    continue
                if not any(
                    _target_matches_action(target, action)
                    or _target_matches_action_target_id(target, action)
                    for target in exact_location_receptacle_targets
                ):
                    continue
                base = _target_physical_base(action.get("target"))
                if base:
                    location_receptacle_physical_bases.add(base)
        active_location_receptacle_targets = (
            exact_location_receptacle_targets
            if location_receptacle_physical_bases
            else location_receptacle_targets
        )

        def action_matches_location_receptacle_target(action: dict) -> bool:
            if any(
                _target_matches_action(target, action)
                or _target_matches_action_target_id(target, action)
                or _target_matches_action_child_receptacle(target, action)
                for target in active_location_receptacle_targets
            ):
                return True
            action_base = _target_physical_base(action.get("target"))
            return bool(
                action_base and action_base in location_receptacle_physical_bases
            )

        def action_matches_location_parent(action: dict) -> bool:
            if any(
                _target_matches_action_parent(target, action)
                or _target_matches_action_parent_child_receptacle(target, action)
                for target in active_location_receptacle_targets
            ):
                return True
            parent_base = _target_physical_base(action.get("parent_target"))
            return bool(
                parent_base and parent_base in location_receptacle_physical_bases
            )

        holding = _agent_is_holding(obs)
        placement_goal_active = bool(
            holding
            and _instruction_is_placement_goal(self.instruction)
            and _holding_placement_progress_available(actions)
        )
        if not holding:
            inferred_task_pickup_targets = _instruction_pickup_target_families(
                self.instruction, actions
            ) | _instruction_retrieval_target_families(self.instruction, actions)
            explicit_task_pickup_target = _target_family(
                obs.get("task_object_label") or obs.get("task_object_type")
            )
            if explicit_task_pickup_target and any(
                action.get("action_type") in {"Navigate", "PickUp"}
                and _target_matches_action(explicit_task_pickup_target, action)
                for action in actions
            ):
                task_pickup_targets = {explicit_task_pickup_target}
            else:
                task_pickup_targets = inferred_task_pickup_targets
        else:
            task_pickup_targets = set()
        task_pickup_guard_active = bool(task_pickup_targets)
        retrieval_pickup_goal_active = bool(
            task_pickup_targets and _instruction_is_retrieval_goal(self.instruction)
        )
        task_pickup_object_action_available = bool(
            task_pickup_targets
            and any(
                action.get("action_type") in {"Navigate", "PickUp"}
                and any(
                    _target_matches_action(target, action)
                    for target in task_pickup_targets
                )
                for action in actions
            )
        )
        task_pickup_action_available = bool(
            _instruction_is_placement_goal(self.instruction)
            and task_pickup_object_action_available
        )
        task_pickup_parent_targets = {
            str(value)
            for action in actions
            if action.get("action_type") in {"Navigate", "PickUp"}
            and any(
                _target_matches_action(target, action) for target in task_pickup_targets
            )
            for value in (
                action.get("parent_target"),
                action.get("parent_target_label"),
            )
            if value
        }
        task_parent_open_available = bool(
            task_pickup_action_available
            and task_pickup_parent_targets
            and any(
                action.get("action_type") == "Open"
                and any(
                    _target_matches_action(target, action)
                    or _target_matches_action_target_id(target, action)
                    for target in task_pickup_parent_targets
                )
                for action in actions
            )
        )
        exact_location_object_available = bool(
            not holding
            and location_exact_object_targets
            and any(
                action.get("action_type") in {"Navigate", "PickUp"}
                and any(
                    _target_matches_action(target, action)
                    or _target_matches_action_target_id(target, action)
                    for target in location_exact_object_targets
                )
                and (
                    not location_receptacle_targets
                    or not (
                        action.get("parent_target")
                        or action.get("parent_target_label")
                        or action.get("parent_target_type")
                    )
                    or action_matches_location_parent(action)
                )
                for action in actions
            )
        )
        if exact_location_object_available:
            location_object_targets = location_exact_object_targets
        location_pickup_available = bool(
            not holding
            and location_object_targets
            and any(
                action.get("action_type") == "PickUp"
                and any(
                    _target_matches_action(target, action)
                    or _target_matches_action_target_id(target, action)
                    for target in location_object_targets
                )
                for action in actions
            )
        )
        location_parent_matched_pickup_available = bool(
            location_pickup_available
            and location_receptacle_targets
            and any(
                action.get("action_type") == "PickUp"
                and any(
                    _target_matches_action(target, action)
                    or _target_matches_action_target_id(target, action)
                    for target in location_object_targets
                )
                and action_matches_location_parent(action)
                for action in actions
            )
        )
        location_parent_matched_object_nav_available = bool(
            not holding
            and location_object_targets
            and location_receptacle_targets
            and any(
                action.get("action_type") == "Navigate"
                and any(
                    _target_matches_action(target, action)
                    or _target_matches_action_target_id(target, action)
                    for target in location_object_targets
                )
                and action_matches_location_parent(action)
                for action in actions
            )
        )
        location_object_nav_available = bool(
            not holding
            and location_object_targets
            and any(
                action.get("action_type") == "Navigate"
                and any(
                    _target_matches_action(target, action)
                    or _target_matches_action_target_id(target, action)
                    for target in location_object_targets
                )
                for action in actions
            )
        )
        location_object_parent_metadata_available = bool(
            location_receptacle_targets
            and any(
                action.get("action_type") in {"Navigate", "PickUp"}
                and any(
                    _target_matches_action(target, action)
                    or _target_matches_action_target_id(target, action)
                    for target in location_object_targets
                )
                and (
                    action.get("parent_target")
                    or action.get("parent_target_label")
                    or action.get("parent_target_type")
                )
                for action in actions
            )
        )
        location_object_action_targets = {
            _target_exact(action.get("target"))
            for action in actions
            if action.get("action_type") in {"Navigate", "PickUp"}
            and any(
                _target_matches_action(target, action)
                or _target_matches_action_target_id(target, action)
                for target in location_object_targets
            )
            and _target_exact(action.get("target"))
        }
        allow_unique_location_object_parent_mismatch = bool(
            not holding
            and location_receptacle_targets
            and getattr(self, "_last_task_progress", 0.0) >= 0.5
            and len(location_object_action_targets) == 1
        )
        location_receptacle_nav_available = bool(
            not holding
            and location_receptacle_targets
            and any(
                action.get("action_type") == "Navigate"
                and action_matches_location_receptacle_target(action)
                for action in actions
            )
        )
        strict_location_parent_active = bool(
            location_receptacle_targets
            and location_object_parent_metadata_available
            and (
                getattr(self, "_last_task_progress", 0.0) >= 0.5
                or location_parent_matched_pickup_available
                or location_parent_matched_object_nav_available
                or location_receptacle_nav_available
            )
        )
        location_parent_compatible_object_action_available = bool(
            location_object_targets
            and any(
                action.get("action_type") in {"Navigate", "PickUp"}
                and any(
                    _target_matches_action(target, action)
                    or _target_matches_action_target_id(target, action)
                    for target in location_object_targets
                )
                and (
                    not strict_location_parent_active
                    or action_matches_location_parent(action)
                    or (
                        allow_unique_location_object_parent_mismatch
                        and _target_exact(action.get("target"))
                        in location_object_action_targets
                    )
                )
                for action in actions
            )
        )
        location_preferred_available = bool(
            location_pickup_available
            or location_object_nav_available
            or location_receptacle_nav_available
        )
        preferred_physical_put_available = bool(
            preferred_physical_put_targets
            and any(
                action.get("action_type") == "PutObject"
                and any(
                    _target_matches_action_target_id(target, action)
                    for target in preferred_physical_put_targets
                )
                for action in actions
            )
        )
        preferred_physical_nav_available = bool(
            holding
            and preferred_physical_put_targets
            and any(
                action.get("action_type") == "Navigate"
                and any(
                    _target_matches_action_target_id(target, action)
                    for target in preferred_physical_put_targets
                )
                for action in actions
            )
        )
        preferred_physical_open_available = bool(
            holding
            and preferred_physical_put_targets
            and any(
                action.get("action_type") == "Open"
                and any(
                    _target_matches_action_target_id(target, action)
                    for target in preferred_physical_put_targets
                )
                for action in actions
            )
        )
        preferred_put_available = bool(
            not preferred_physical_put_available
            and preferred_put_targets
            and any(
                action.get("action_type") == "PutObject"
                and any(
                    _target_matches_action(target, action)
                    for target in preferred_put_targets
                )
                for action in actions
            )
        )
        preferred_nav_available = bool(
            holding
            and not preferred_physical_nav_available
            and preferred_put_targets
            and any(
                action.get("action_type") == "Navigate"
                and any(
                    _target_matches_action(target, action)
                    for target in preferred_put_targets
                )
                for action in actions
            )
        )
        preferred_open_available = bool(
            holding
            and not preferred_physical_open_available
            and preferred_put_targets
            and any(
                action.get("action_type") == "Open"
                and any(
                    _target_matches_action(target, action)
                    for target in preferred_put_targets
                )
                for action in actions
            )
        )
        preferred_memory_action_available = holding and _has_preferred_memory_action(
            actions,
            preferred_put_targets=preferred_put_targets,
            preferred_physical_put_targets=preferred_physical_put_targets,
        )
        semantic_alternative_available = bool(
            holding
            and deprioritized_physical_put_targets
            and preferred_put_targets
            and any(
                action.get("action_type") in {"Navigate", "PutObject"}
                and any(
                    _target_matches_action(target, action)
                    for target in preferred_put_targets
                )
                and not any(
                    _target_matches_action_target_id(target, action)
                    or _target_matches_action(target, action)
                    for target in deprioritized_physical_put_targets
                )
                for action in actions
            )
        )
        preferred_action_available = bool(
            preferred_physical_put_available
            or preferred_physical_nav_available
            or preferred_physical_open_available
            or preferred_put_available
            or preferred_nav_available
            or preferred_open_available
            or preferred_memory_action_available
        )
        storage_goal_filter_active = bool(
            holding
            and _instruction_requires_openable_storage(self.instruction)
            and not preferred_put_targets
            and not preferred_physical_put_targets
            and any(
                action.get("action_type") in {"Navigate", "Open", "PutObject"}
                and _action_matches_failure_storage_target(action)
                for action in actions
            )
        )
        storage_forward_candidates = [
            action
            for action in actions
            if action.get("action_type") in {"Navigate", "Open", "PutObject"}
            and _action_matches_failure_storage_target(action)
            and not action_is_blocked(action)
        ]
        placement_goal_incomplete_active = bool(
            _instruction_is_placement_goal(self.instruction)
            and getattr(self, "_last_task_progress", 0.0) < 0.999
            and any(
                action.get("action_type") in {"Navigate", "PickUp", "Open", "PutObject"}
                and not action_is_blocked(action)
                for action in actions
            )
        )
        if (
            not blocked
            and not preferred_physical_put_available
            and not preferred_put_available
            and not preferred_physical_nav_available
            and not preferred_physical_open_available
            and not preferred_nav_available
            and not preferred_open_available
            and not preferred_memory_action_available
            and not semantic_alternative_available
            and not location_preferred_available
            and not placement_goal_active
            and not task_pickup_guard_active
            and not storage_goal_filter_active
            and not task_parent_open_available
            and not retrieval_pickup_goal_active
            and not placement_goal_incomplete_active
        ):
            return obs
        filtered = []
        for action in actions:
            action_type = action.get("action_type")
            matches_semantic_preferred = any(
                _target_matches_action(target, action)
                for target in preferred_put_targets
            )
            matches_physical_preferred = any(
                _target_matches_action_target_id(target, action)
                for target in preferred_physical_put_targets
            )
            matches_any_preferred = (
                matches_semantic_preferred or matches_physical_preferred
            )
            matches_location_object = any(
                _target_matches_action(target, action)
                or _target_matches_action_target_id(target, action)
                for target in location_object_targets
            )
            matches_location_receptacle = any(
                _target_matches_action(target, action)
                or _target_matches_action_target_id(target, action)
                or _target_matches_action_child_receptacle(target, action)
                or _target_matches_action_parent(target, action)
                or _target_matches_action_parent_child_receptacle(target, action)
                for target in active_location_receptacle_targets
            )
            matches_location_receptacle_target = (
                action_matches_location_receptacle_target(action)
            )
            matches_location_parent = action_matches_location_parent(action)
            matches_location_receptacle = (
                matches_location_receptacle
                or matches_location_receptacle_target
                or matches_location_parent
            )
            matches_task_pickup = any(
                _target_matches_action(target, action) for target in task_pickup_targets
            )
            matches_task_pickup_parent = any(
                _target_matches_action(target, action)
                or _target_matches_action_target_id(target, action)
                for target in task_pickup_parent_targets
            )
            matches_unique_location_object = bool(
                allow_unique_location_object_parent_mismatch
                and matches_location_object
                and _target_exact(action.get("target"))
                in location_object_action_targets
            )
            matches_failure_storage = _action_matches_failure_storage_target(action)
            if action_is_blocked(action):
                continue
            if placement_goal_incomplete_active and action_type in {"Done", "Stop"}:
                continue
            if task_parent_open_available and action_type not in {
                "Navigate",
                "PickUp",
                "Open",
            }:
                continue
            if (
                task_parent_open_available
                and action_type == "Open"
                and not matches_task_pickup_parent
            ):
                continue
            if (
                task_pickup_action_available
                and action_type in {"Navigate", "PickUp"}
                and not matches_task_pickup
            ):
                continue
            if (
                task_pickup_guard_active
                and action_type == "PickUp"
                and not matches_task_pickup
            ):
                continue
            if retrieval_pickup_goal_active and action_type in {"Done", "Stop"}:
                continue
            if (
                retrieval_pickup_goal_active
                and matches_task_pickup
                and action_type in {"Open", "Close", "TransferContents", "PutObject"}
            ):
                continue
            if (
                retrieval_pickup_goal_active
                and task_pickup_object_action_available
                and not location_preferred_available
                and action_type in {"Navigate", "PickUp"}
                and not matches_task_pickup
            ):
                continue
            if placement_goal_active and action_type not in {
                "Navigate",
                "Open",
                "PutObject",
            }:
                continue
            if (
                storage_goal_filter_active
                and action_type in {"Navigate", "Open", "PutObject"}
                and not matches_failure_storage
            ):
                continue
            if task_pickup_guard_active and action_type in {"Done", "Stop"}:
                continue
            if location_preferred_available and action_type not in {
                "Navigate",
                "PickUp",
                "Open",
            }:
                continue
            if (
                location_preferred_available
                and action_type == "Open"
                and not matches_location_receptacle_target
            ):
                continue
            if location_pickup_available:
                if (
                    action_type in {"Navigate", "Open"}
                    and matches_location_receptacle_target
                    and not location_parent_compatible_object_action_available
                ):
                    filtered.append(action)
                    continue
                if action_type not in {"Navigate", "PickUp"}:
                    continue
                if not matches_location_object:
                    continue
                if (
                    strict_location_parent_active
                    and action_type == "PickUp"
                    and not matches_location_parent
                    and not matches_unique_location_object
                ):
                    continue
                if (
                    strict_location_parent_active
                    and action_type == "Navigate"
                    and not matches_location_parent
                    and not matches_unique_location_object
                ):
                    continue
            elif location_preferred_available and action_type == "Navigate":
                object_nav_allowed = matches_location_object
                if (
                    strict_location_parent_active
                    and not matches_location_parent
                    and not matches_unique_location_object
                ):
                    object_nav_allowed = False
                if not (object_nav_allowed or matches_location_receptacle_target):
                    continue
            elif (
                location_preferred_available
                and action_type == "PickUp"
                and not matches_location_object
            ):
                continue
            elif location_preferred_available and action_type in {"Done", "Stop"}:
                continue
            if (
                holding
                and preferred_put_targets
                and action_type in {"Close", "TransferContents"}
            ):
                continue
            if (
                holding
                and preferred_put_targets
                and action_type == "PutObject"
                and not matches_any_preferred
            ):
                continue
            if (
                preferred_action_available
                and action_type in {"Navigate", "Open", "PutObject"}
                and not matches_any_preferred
            ):
                continue
            if (
                semantic_alternative_available
                and action_type in {"Navigate", "PutObject"}
                and any(
                    _target_matches_action_target_id(target, action)
                    or _target_matches_action(target, action)
                    for target in deprioritized_physical_put_targets
                )
            ):
                continue
            if (
                preferred_physical_nav_available
                and action_type == "Navigate"
                and not any(
                    _target_matches_action_target_id(target, action)
                    for target in preferred_physical_put_targets
                )
            ):
                continue
            if (
                preferred_physical_open_available
                and action_type == "Open"
                and not any(
                    _target_matches_action_target_id(target, action)
                    for target in preferred_physical_put_targets
                )
            ):
                continue
            if (
                preferred_physical_put_available
                and action_type == "PutObject"
                and not any(
                    _target_matches_action_target_id(target, action)
                    for target in preferred_physical_put_targets
                )
            ):
                continue
            if (
                preferred_nav_available
                and action_type == "Navigate"
                and not any(
                    _target_matches_action(target, action)
                    for target in preferred_put_targets
                )
            ):
                continue
            if (
                (preferred_nav_available or preferred_open_available)
                and action_type in {"Navigate", "Open"}
                and not any(
                    _target_matches_action(target, action)
                    for target in preferred_put_targets
                )
            ):
                continue
            if (
                preferred_open_available
                and action_type == "Open"
                and not any(
                    _target_matches_action(target, action)
                    for target in preferred_put_targets
                )
            ):
                continue
            if (
                (preferred_nav_available or preferred_open_available)
                and action_type in {"PutObject", "TransferContents"}
                and not any(
                    _target_matches_action(target, action)
                    for target in preferred_put_targets
                )
            ):
                continue
            if (
                preferred_put_available
                and action_type == "PutObject"
                and not any(
                    _target_matches_action(target, action)
                    for target in preferred_put_targets
                )
            ):
                continue
            if preferred_memory_action_available and action_type in {"Done", "Stop"}:
                continue
            filtered.append(action)
        if storage_goal_filter_active and storage_forward_candidates:
            filtered_storage = [
                action for action in filtered if action in storage_forward_candidates
            ]
            filtered = filtered_storage or storage_forward_candidates
        if not filtered:
            return obs
        if len(filtered) == len(actions):
            return obs
        next_obs = dict(obs)
        next_obs["available_actions"] = filtered
        return next_obs

    def build_context(self) -> tuple[list[dict], dict[str, Any]]:
        memory_text = _build_retrieved_memory_text(
            self.memory,
            self.format_query_results,
            self.instruction,
            self.contexts,
            self.current_scene,
            self.current_namespace,
            event_top_k=self.event_top_k,
            spatial_top_k=self.spatial_top_k,
            scene_top_k=self.scene_top_k,
        )
        online_constraints = self.memory.query_constraints(
            "avoid failed invalid action placement",
            top_k=3,
            namespace=self.current_namespace,
        )
        if online_constraints:
            constraint_text = (
                "## embodied_memorizer Online Interaction Constraints\n"
                + self.format_query_results(online_constraints)
            )
            memory_text = (
                f"{memory_text}\n\n{constraint_text}"
                if memory_text.strip()
                else constraint_text
            )
        if not memory_text.strip():
            memory_text = _safe_build_memory_text(
                self.build_memory_context,
                self.memory,
                self.config,
                self.instruction,
            )
        if not memory_text.strip():
            memory_text = "embodied_memorizer did not retrieve relevant past memory for this probe."
        stats = self.memory.get_stats()
        stats.update(
            {
                "backend": "embodied_memorizer.MemorySystem",
                "memory_module_root": self.memory_module_root,
                "embedding_model": self.embedding_model,
                "embedding_device": self.embedding_device,
                "online_step_count": self.online_step_count,
                "action_filtering_enabled": bool(
                    getattr(self, "enable_action_filtering", True)
                ),
                "online_writeback_enabled": bool(
                    getattr(self, "enable_online_writeback", True)
                ),
                "blocked_action_count": len(self.blocked_action_signatures),
                "memory_blocked_action_count": len(
                    self.memory_blocked_action_signatures
                ),
                "memory_habit_blocked_action_count": len(
                    self.memory_habit_blocked_action_signatures
                ),
                "memory_preferred_put_target_count": len(
                    self.memory_preferred_put_targets
                ),
                "memory_preferred_physical_put_target_count": len(
                    self.memory_preferred_physical_put_targets
                ),
                "memory_deprioritized_physical_put_target_count": len(
                    self.memory_deprioritized_physical_put_targets
                ),
                "memory_location_object_target_count": len(
                    self.memory_location_object_targets
                ),
                "memory_location_exact_object_target_count": len(
                    self.memory_location_exact_object_targets
                ),
                "memory_location_receptacle_target_count": len(
                    self.memory_location_receptacle_targets
                ),
            }
        )
        return _context_from_memory_text(memory_text, stats), stats


def build_memory_module_context(
    contexts: list[dict],
    instruction: str,
    *,
    object_labels: dict[str, str] | None = None,
    current_scene: Any = None,
    current_namespace: str | None = None,
    memory_module_root: str | None = None,
    embedding_model: str | None = None,
    embedding_device: str = "cpu",
    max_memory_tokens: int = 1600,
    experience_top_k: int = 5,
    event_top_k: int = 14,
    spatial_top_k: int = 12,
    scene_top_k: int = 3,
) -> tuple[list[dict], dict[str, Any]]:
    """Return planner-visible context retrieved by the real embodied_memorizer."""
    (
        MemorySystem,
        DEFAULT_CONFIG,
        merge_config,
        build_memory_context,
        format_query_results,
    ) = _ensure_memory_module(memory_module_root)
    if embedding_model is None and Path(DEFAULT_LOCAL_EMBEDDING_MODEL).exists():
        embedding_model = DEFAULT_LOCAL_EMBEDDING_MODEL
    config = _memory_config(
        default_config=DEFAULT_CONFIG,
        merge_config=merge_config,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
        max_memory_tokens=max_memory_tokens,
        event_top_k=event_top_k,
        spatial_top_k=spatial_top_k,
        scene_top_k=scene_top_k,
    )
    memory = MemorySystem(config)
    _preload_contexts(memory, contexts, object_labels=object_labels)
    memory_text = _build_retrieved_memory_text(
        memory,
        format_query_results,
        instruction,
        contexts,
        current_scene,
        current_namespace,
        experience_top_k=experience_top_k,
        event_top_k=event_top_k,
        spatial_top_k=spatial_top_k,
        scene_top_k=scene_top_k,
    )
    if not memory_text.strip():
        default_memory_text = _safe_build_memory_text(
            build_memory_context, memory, config, instruction
        )
        memory_text = default_memory_text
    if not memory_text.strip():
        memory_text = (
            "embodied_memorizer did not retrieve relevant past memory for this probe."
        )

    stats = memory.get_stats()
    stats.update(
        {
            "backend": "embodied_memorizer.MemorySystem",
            "memory_module_root": str(
                Path(
                    memory_module_root
                    or os.getenv("MEMORY_MODULE_ROOT")
                    or DEFAULT_MEMORY_MODULE_ROOT
                )
            ),
            "embedding_model": embedding_model or DEFAULT_CONFIG.get("embedding_model"),
            "embedding_device": embedding_device,
        }
    )
    return _context_from_memory_text(memory_text, stats), stats
