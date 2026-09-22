"""Retained protocol helpers from the original experiment code."""

from __future__ import annotations


import argparse


import json


import re


from copy import deepcopy


from typing import Any


from jsonschema import Draft7Validator


from emem_bench.evaluation.env import DEFAULT_ACTION_TYPES, EmbodiedMemorizerEnv


from emem_bench.evaluation.matchers import object_type


from emem_bench.evaluation.planner import (
    compact_available_actions,
    compact_held_object,
    compact_semantic_available_actions,
    compact_visible_objects,
    strip_object_ids,
)


from emem_bench.evaluation.memory_adapter import MemoryRuntime


from emem_bench.evaluation.memory_adapter import (
    _action_text,
    _object_label,
    _visible_object_names,
)


from emem_bench.evaluation.grounding import (
    bind_action_to_observation_space,
    bind_sequence_action_to_observation_space,
    repair_action_id_from_structured_prediction,
    short_error,
)


from emem_bench.construction.scene_utils import load_scene_metadata


TWO_PHASE_ROLLOUT_SYSTEM_PROMPT = """You are an expert embodied-memory agent controlling a household robot.
A valid record has two ordered stages:
1. context_ingestion: replay prior model-visible context observations and call
   real embodied_memorizer update tools to store durable memory.
2. probe_rollout: for each probe turn, first run memory_phase with real
   embodied_memorizer tools, then run action_phase with one action_sequence after
   tool results are available.

Never put action_sequence in memory_phase. Never put embodied_memorizer tools in
action_phase. Never treat prior context as an oracle prompt paragraph; prior
context must enter the memory state through embodied_memorizer tool calls."""


ALIGNED_CONTEXT_INGESTION_GUIDANCE = """ALIGNED CONTEXT INGESTION CONTRACT:
- The observation is normal prior embodied experience, never a future probe label.
- If new_visible_relations is nonempty, write every listed subject relation with update_object and also record one successful remember_event phrased "SUBJECT was observed on/in TARGET" so both SpatialMemory and ExperienceMemory retain it.
- Preserve the normalized durable class in failed interaction feedback, such as locked or invalid placement, together with the exact action target.
- Use update tools only. Retrieval tools and robot actions are forbidden."""


MEMORYMODULE_UPDATE_TOOLS = frozenset(
    {
        "remember_object",
        "update_object",
        "remember_event",
        "remember_scene",
    }
)


MEMORYMODULE_RETRIEVE_TOOLS = frozenset(
    {
        "query_spatial",
        "query_event",
        "query_scene",
        "query_experience",
    }
)


MEMORYMODULE_MEMORY_TOOLS = MEMORYMODULE_UPDATE_TOOLS | MEMORYMODULE_RETRIEVE_TOOLS


CONCRETE_SEQUENCE_ACTION_TYPES = frozenset(DEFAULT_ACTION_TYPES) - {"Done", "Stop"}


def _json_schema_response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def action_phase_response_format(
    *,
    min_actions: int = 1,
    max_actions: int | None = None,
    available_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if min_actions < 1:
        raise ValueError("min_actions must be >= 1")
    if max_actions is not None and max_actions < 1:
        raise ValueError("max_actions must be >= 1")
    if max_actions is not None and min_actions > max_actions:
        raise ValueError("min_actions must be <= max_actions")
    response_format = _json_schema_response_format(
        "action_phase_closed_loop_v3" if max_actions == 1 else "action_phase_v3",
        {
            "type": "object",
            "properties": {
                "phase": {"type": "string", "const": "action"},
                "reasoning": {
                    "type": "object",
                    "properties": {
                        "memory_evidence": {"type": "string"},
                        "planning": {"type": "string"},
                        "sequence_check": {"type": "string"},
                    },
                    "required": ["memory_evidence", "planning", "sequence_check"],
                    "additionalProperties": False,
                },
                "tool_calls": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool_name": {
                                "type": "string",
                                "const": "action_sequence",
                            },
                            "arguments": {
                                "type": "object",
                                "properties": {
                                    "actions": {
                                        "type": "array",
                                        "minItems": min_actions,
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "action_type": {
                                                    "type": "string",
                                                    "enum": sorted(
                                                        CONCRETE_SEQUENCE_ACTION_TYPES
                                                    ),
                                                },
                                                "target_label": {
                                                    "type": ["string", "null"],
                                                },
                                                "target_type": {
                                                    "type": ["string", "null"],
                                                },
                                            },
                                            "required": [
                                                "action_type",
                                                "target_label",
                                                "target_type",
                                            ],
                                            "additionalProperties": False,
                                        },
                                    },
                                },
                                "required": ["actions"],
                                "additionalProperties": False,
                            },
                        },
                        "required": ["tool_name", "arguments"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["phase", "reasoning", "tool_calls"],
            "additionalProperties": False,
        },
    )
    if max_actions is not None:
        response_format["json_schema"]["schema"]["properties"]["tool_calls"]["items"][
            "properties"
        ]["arguments"]["properties"]["actions"]["maxItems"] = max_actions
    exact_actions: list[dict[str, Any]] = []
    seen_actions: set[tuple[str, str | None, str | None]] = set()
    for action in available_actions or []:
        if not isinstance(action, dict):
            continue
        action_type = str(action.get("action_type") or "")
        if action_type not in CONCRETE_SEQUENCE_ACTION_TYPES:
            continue
        target_label = action.get("target_label")
        target_type = action.get("target_type")
        if target_label is not None:
            target_label = str(target_label)
        if target_type is not None:
            target_type = str(target_type)
        signature = (action_type, target_label, target_type)
        if signature in seen_actions:
            continue
        seen_actions.add(signature)
        exact_actions.append(
            {
                "type": "object",
                "properties": {
                    "action_type": {"const": action_type},
                    "target_label": {"const": target_label},
                    "target_type": {"const": target_type},
                },
                "required": ["action_type", "target_label", "target_type"],
                "additionalProperties": False,
            }
        )
    if exact_actions:
        response_format["json_schema"]["schema"]["properties"]["tool_calls"]["items"][
            "properties"
        ]["arguments"]["properties"]["actions"]["items"] = {
            "anyOf": exact_actions,
        }
    return response_format


def normalize_memorymodule_tool_calls(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept common tool-call spellings and store the project schema."""
    tool_calls = payload.get("tool_calls")
    if not isinstance(tool_calls, list):
        return payload
    normalized = []
    changed = False
    for call in tool_calls:
        if not isinstance(call, dict):
            normalized.append(call)
            continue
        if "tool" in call and "args" in call:
            normalized.append(call)
            continue
        name = call.get("tool") or call.get("name") or call.get("tool_name")
        args = call.get("args")
        if not isinstance(args, dict):
            args = call.get("arguments")
        if name in MEMORYMODULE_MEMORY_TOOLS and isinstance(args, dict):
            next_call = dict(call)
            next_call["tool"] = name
            next_call["args"] = args
            next_call.pop("name", None)
            next_call.pop("tool_name", None)
            next_call.pop("arguments", None)
            normalized.append(next_call)
            changed = True
        else:
            normalized.append(call)
    if changed:
        payload = dict(payload)
        payload["tool_calls"] = normalized
    return payload


def context_room_name(session: dict[str, Any]) -> str:
    for key in ("room_type", "scene"):
        value = session.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    name = str(session.get("session_name") or "")
    match = re.match(r"(?P<room>[A-Za-z][A-Za-z0-9_]*)", name)
    if match:
        return match.group("room")
    return "household"


def english_context_description(description: Any, room: str) -> str:
    text = str(description or "")
    if not re.search(r"[\u4e00-\u9fff]", text):
        return text
    if "锁" in text or "失败" in text:
        return f"Past context task execution in the {room}, including a failed interaction."
    if "搬" in text or "移动" in text:
        return f"Past context task execution in the {room}, including an object relocation."
    if "视野" in text or "扫过" in text:
        return f"Past context task execution in the {room}, including passive object observations."
    return f"Past context task execution in the {room}."


def english_memory_cue_description(description: Any) -> str:
    text = str(description or "")
    if not re.search(r"[\u4e00-\u9fff]", text):
        return text
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]*", text)
    if "失败" in text and words:
        return f"Observed failed interaction involving {' '.join(words[:2])}."
    if ("搬" in text or "移动" in text) and words:
        return f"Observed object relocation involving {' '.join(words[:3])}."
    if ("视野" in text or "出现" in text) and words:
        return f"Observed visible object context involving {' '.join(words[:3])}."
    return "Observed a relevant context memory cue."


def contexts_with_english_metadata(
    contexts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized = json.loads(json.dumps(contexts, ensure_ascii=False))
    for idx, session in enumerate(normalized):
        if not isinstance(session, dict):
            continue
        room = context_room_name(session)
        raw_name = str(session.get("session_name") or "")
        if re.search(r"[\u4e00-\u9fff]", raw_name):
            if "Noise" in raw_name or "无关" in raw_name:
                session["session_name"] = f"{room} distractor task execution"
            elif "失败" in raw_name:
                session["session_name"] = f"{room} context task with failure event"
            else:
                session["session_name"] = f"{room} context task execution"
        elif not raw_name:
            session["session_name"] = f"context_session_{idx}"
        session["description"] = english_context_description(
            session.get("description"), room
        )
        for step in session.get("steps") or []:
            if not isinstance(step, dict):
                continue
            cue = step.get("memory_cue_exposed")
            if isinstance(cue, dict):
                cue["description"] = english_memory_cue_description(
                    cue.get("description")
                )
    return normalized


def memorymodule_tool_schemas(
    registry,
) -> list[dict[str, Any]]:
    schemas = []
    for schema in registry.get_schemas():
        name = (schema.get("function") or {}).get("name")
        if name in MEMORYMODULE_MEMORY_TOOLS:
            schemas.append(schema)
    return schemas


def context_ingestion_tool_schemas(
    registry,
) -> list[dict[str, Any]]:
    """Return the exact registered schemas for model-controlled memory writes."""
    return [
        schema
        for schema in memorymodule_tool_schemas(registry)
        if (schema.get("function") or {}).get("name") in MEMORYMODULE_UPDATE_TOOLS
    ]


def _closed_declared_object_schemas(value: Any) -> Any:
    """Close objects with declared properties while preserving free-form maps."""
    if isinstance(value, list):
        return [_closed_declared_object_schemas(item) for item in value]
    if not isinstance(value, dict):
        return value
    closed = {key: _closed_declared_object_schemas(item) for key, item in value.items()}
    if closed.get("type") == "object" and isinstance(closed.get("properties"), dict):
        closed["additionalProperties"] = False
    return closed


def memorymodule_tool_schema_map(
    tool_schemas: list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for schema in tool_schemas or []:
        function = schema.get("function") if isinstance(schema, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name:
            mapped[name] = schema
    return mapped


def validate_memorymodule_tool_call(
    call: dict[str, Any],
    *,
    tool_schemas: list[dict[str, Any]] | None,
) -> list[str]:
    """Validate one normalized call against the registry's executable signature."""
    name = call.get("tool")
    args = call.get("args")
    if not isinstance(name, str) or not name:
        return ["missing_tool_name"]
    if not isinstance(args, dict):
        return [f"{name}_missing_args"]
    schema = memorymodule_tool_schema_map(tool_schemas).get(name)
    if schema is None:
        return [f"missing_registered_schema:{name}"] if tool_schemas is not None else []
    parameters = deepcopy((schema.get("function") or {}).get("parameters") or {})
    parameters = _closed_declared_object_schemas(parameters)
    errors = []
    for error in sorted(
        Draft7Validator(parameters).iter_errors(args), key=lambda item: list(item.path)
    ):
        path = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"{name}_args_schema:{path}:{error.message}")
    return errors


def build_context_ingestion_system_prompt(
    registry,
) -> str:
    """Expose the same update signatures that the registry will execute."""
    schemas = context_ingestion_tool_schemas(registry)
    return (
        TWO_PHASE_ROLLOUT_SYSTEM_PROMPT
        + "\n\nCONTEXT INGESTION TOOL SIGNATURES:\n"
        + json.dumps(schemas, ensure_ascii=False, indent=2)
        + "\nUse exactly these argument names and types. Do not invent aliases."
    )


def validate_action_phase_payload(
    payload: dict[str, Any],
    *,
    min_actions: int = 1,
    max_actions: int | None = None,
    available_actions: list[dict[str, Any]] | None = None,
) -> tuple[bool, list[str]]:
    errors = []
    if payload.get("phase") != "action":
        errors.append("missing_or_wrong_action_phase")
    tool_calls = payload.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return False, errors + ["missing_tool_calls"]
    if len(tool_calls) != 1:
        errors.append(f"action_phase_wrong_tool_call_count:{len(tool_calls)}")
    names = [call.get("tool_name") for call in tool_calls if isinstance(call, dict)]
    if names.count("action_sequence") != 1:
        errors.append("missing_or_duplicate_action_sequence")
    serialized_actions: list[dict[str, Any]] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            errors.append("non_dict_tool_call")
            continue
        name = call.get("tool_name")
        if name != "action_sequence":
            errors.append(f"action_phase_forbidden_tool:{name}")
            continue
        if not isinstance(call.get("arguments"), dict):
            errors.append("action_sequence_missing_arguments")
            continue
        actions = (call.get("arguments") or {}).get("actions")
        if not isinstance(actions, list) or not actions:
            errors.append("empty_action_sequence")
            continue
        serialized_actions.extend(
            action for action in actions if isinstance(action, dict)
        )
        for idx, action in enumerate(actions):
            if not isinstance(action, dict):
                errors.append(f"non_dict_action:{idx}")
                continue
            action_type = action.get("action_type")
            if not action_type:
                errors.append(f"missing_action_type:{idx}")
            elif action_type not in CONCRETE_SEQUENCE_ACTION_TYPES:
                errors.append(f"noncanonical_action_type:{idx}:{action_type}")
    if serialized_actions and len(serialized_actions) < min_actions:
        errors.append(
            f"action_sequence_too_short:{len(serialized_actions)}<{min_actions}"
        )
    if max_actions is not None and len(serialized_actions) > max_actions:
        if max_actions == 1 and available_actions is not None:
            errors.append(f"closed_loop_action_count:{len(serialized_actions)}")
        else:
            errors.append(
                f"action_sequence_too_long:{len(serialized_actions)}>{max_actions}"
            )
    if available_actions is not None and len(serialized_actions) == 1:
        planned = serialized_actions[0]
        planned_signature = (
            planned.get("action_type"),
            planned.get("target_label"),
            planned.get("target_type"),
        )
        available_signatures = {
            (
                action.get("action_type"),
                action.get("target_label"),
                action.get("target_type"),
            )
            for action in available_actions
            if isinstance(action, dict)
        }
        if planned_signature not in available_signatures:
            errors.append(
                "closed_loop_action_not_available:"
                + ":".join(
                    str(value) if value is not None else "None"
                    for value in planned_signature
                )
            )
    return not errors, errors


def compact_rollout_observation(
    obs: dict[str, Any],
    *,
    max_context_steps: int,
    max_visible_objects: int,
    max_navigate_actions: int | None = None,
) -> dict[str, Any]:
    visible_objects = compact_visible_objects(obs, max_visible_objects)
    observation = {
        "instruction": obs.get("instruction"),
        "memory_system": {
            "backend": "embodied_memorizer.MemorySystem",
            "note": (
                "Prior context was written into embodied_memorizer through the "
                "context_ingestion tool-call trace in this record. Use "
                "embodied_memorizer query tools to inspect that memory; do not rely "
                "on raw prior context being present in the action prompt."
            ),
        },
        "current_visible_objects": visible_objects,
        "held_object": compact_held_object(obs),
    }
    if obs.get("task_object_label"):
        observation["task_object_label"] = obs.get("task_object_label")
    if obs.get("task_object_type"):
        observation["task_object_type"] = obs.get("task_object_type")
    if max_navigate_actions is None:
        observation["available_actions"] = compact_available_actions(
            obs.get("available_actions", [])
        )
    else:
        actions, summary = compact_semantic_available_actions(
            obs.get("available_actions", []),
            visible_objects=visible_objects,
            max_navigate_actions=max_navigate_actions,
        )
        observation["available_actions"] = actions
        observation["available_action_summary"] = summary
    return observation


def model_visible_action(action: Any) -> Any:
    if not isinstance(action, dict):
        return action
    visible: dict[str, Any] = {}
    for key in (
        "action_id",
        "action_type",
        "target_label",
        "target_type",
        "parent_target_label",
        "parent_target_type",
        "text",
    ):
        value = action.get(key)
        if value is not None:
            visible[key] = value
    if not visible and action.get("action"):
        visible["action_type"] = action.get("action")
    return visible


def model_visible_executed_action(step: dict[str, Any]) -> dict[str, Any]:
    """Expose only environment-visible execution feedback to the model.

    Evaluator-derived completion/progress values remain in the internal step
    record for scoring and logs, but must never enter a subsequent model
    prompt. Otherwise the model can use the hidden oracle as a progress
    signal while replanning or writing memory.
    """
    visible = {
        "sequence_index": step.get("sequence_index"),
        "planned_action": model_visible_action(step.get("planned_action")),
        "executed_action": model_visible_action(step.get("executed_action")),
        "success": bool(step.get("success")),
        "feedback": strip_object_ids(step.get("feedback")),
    }
    if step.get("not_executed"):
        visible["not_executed"] = True
    return visible


def model_visible_executed_actions(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        model_visible_executed_action(step) for step in steps if isinstance(step, dict)
    ]


def action_sequence_from_payload(payload: dict[str, Any]) -> list[Any]:
    for call in payload.get("tool_calls") or []:
        if isinstance(call, dict) and call.get("tool_name") == "action_sequence":
            actions = (call.get("arguments") or {}).get("actions")
            if not isinstance(actions, list):
                return []
            return list(actions)
    return []


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return str(value)


def merge_memorymodule_execution_metadata(
    memory_runtime: MemoryRuntime,
    execution_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        str(key): value
        for key, value in (execution_metadata or {}).items()
        if value is not None
    }
    current_scene = getattr(memory_runtime, "current_scene", None)
    current_namespace = getattr(memory_runtime, "current_namespace", None)
    if current_scene and not metadata.get("scene"):
        metadata["scene"] = current_scene
    metadata["spatial_identity_scope_enabled"] = bool(
        getattr(memory_runtime, "enable_spatial_identity_scope", True)
    )
    if current_namespace:
        metadata["memory_namespace"] = current_namespace
        metadata["source_episode_id"] = current_namespace
    elif metadata.get("memory_namespace") and not metadata.get("source_episode_id"):
        metadata["source_episode_id"] = metadata["memory_namespace"]
    return metadata


def execute_memorymodule_tool_results(
    payload: dict[str, Any],
    *,
    memory_runtime: MemoryRuntime,
    registry,
    execution_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    trusted_metadata = merge_memorymodule_execution_metadata(
        memory_runtime,
        execution_metadata,
    )
    results = []
    registered_schemas = memorymodule_tool_schemas(registry)
    for call in payload.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        name = call.get("tool")
        tool_args = json.loads(json.dumps(call.get("args") or {}, ensure_ascii=False))
        if name not in MEMORYMODULE_MEMORY_TOOLS:
            results.append(
                {
                    "tool": name,
                    "args": tool_args,
                    "status": "error",
                    "success": False,
                    "result": (
                        f"Tool '{name}' is not allowed in memory_phase. "
                        f"Allowed: {', '.join(sorted(MEMORYMODULE_MEMORY_TOOLS))}"
                    ),
                }
            )
            continue
        schema_errors = validate_memorymodule_tool_call(
            call,
            tool_schemas=registered_schemas,
        )
        if schema_errors:
            results.append(
                {
                    "tool": name,
                    "args": tool_args,
                    "status": "error",
                    "success": False,
                    "result": "; ".join(schema_errors),
                }
            )
            continue
        try:
            call_metadata = dict(trusted_metadata)
            if name == "remember_event":
                call_metadata["raw_action_text"] = (
                    tool_args.get("action")
                    or call_metadata.get("raw_action_text")
                    or ""
                )
                call_metadata["raw_feedback_text"] = (
                    tool_args.get("feedback")
                    or call_metadata.get("raw_feedback_text")
                    or ""
                )
            result = registry.execute(
                name,
                json.loads(json.dumps(tool_args, ensure_ascii=False)),
                memory_runtime.memory,
                execution_metadata=call_metadata,
            )
            results.append(
                {
                    "tool": name,
                    "args": tool_args,
                    "status": "ok",
                    "success": True,
                    "result": json_safe(result),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "tool": name,
                    "args": tool_args,
                    "status": "error",
                    "success": False,
                    "result": short_error(exc),
                }
            )
    return results


def context_step_execution_metadata(
    *,
    memory_runtime: MemoryRuntime,
    session_index: int,
    step_index: int,
    session: dict[str, Any],
    step: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    action = step.get("action") or {}
    feedback = step.get("feedback") or {}
    compact_action = observation.get("action") or {}
    compact_feedback = observation.get("feedback") or {}
    metadata = {
        "session_name": observation.get("session_name"),
        "session_index": session_index,
        "step_index": step_index,
        "scene": session.get("scene"),
        "room_type": session.get("room_type") or observation.get("room"),
        "memory_phase": "context_ingestion",
        "raw_action_target": action.get("target"),
        "raw_action_target_type": action.get("action_type") or action.get("action"),
        "raw_action_text": (
            action.get("natural_language")
            or compact_action.get("text")
            or action.get("action_type")
            or action.get("action")
        ),
        "raw_feedback_text": (
            feedback.get("message") or compact_feedback.get("message") or ""
        ),
    }
    return merge_memorymodule_execution_metadata(memory_runtime, metadata)


def context_steps(session: dict[str, Any]) -> list[dict[str, Any]]:
    steps = session.get("steps")
    if isinstance(steps, list):
        return [step for step in steps if isinstance(step, dict)]
    trajectory = session.get("context_trajectory") or {}
    steps = trajectory.get("steps") if isinstance(trajectory, dict) else None
    if isinstance(steps, list):
        return [step for step in steps if isinstance(step, dict)]
    return []


def _non_floor_parent(object_meta: dict[str, Any]) -> str | None:
    for parent_id in object_meta.get("parentReceptacles") or []:
        if (
            parent_id
            and parent_id != "Floor"
            and not str(parent_id).startswith("Floor|")
        ):
            return str(parent_id)
    return None


def _rotate_context_visible_objects(
    *,
    session_index: int,
    step: dict[str, Any],
    max_visible_objects: int,
    ingestion_state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Spend the fixed per-step budget on not-yet-selected session objects first."""
    selected = ingestion_state.setdefault("selected_visible_object_ids", set())
    candidates = [
        obj for obj in (step.get("visible_objects") or []) if isinstance(obj, dict)
    ]

    def selection_key(item: tuple[int, dict[str, Any]]) -> tuple[bool, int]:
        index, obj = item
        object_id = str(obj.get("object_id") or obj.get("objectId") or "")
        return (bool(object_id and (session_index, object_id) in selected), index)

    ranked = sorted(enumerate(candidates), key=selection_key)
    visible = [obj for _, obj in ranked[:max_visible_objects]]
    selected.update(
        (session_index, object_id)
        for obj in visible
        if (object_id := str(obj.get("object_id") or obj.get("objectId") or ""))
    )
    return visible


def static_visible_relation_updates(
    *,
    session_index: int,
    session: dict[str, Any],
    step: dict[str, Any],
    object_labels: dict[str, str] | None,
    context_labels: dict[str, str] | None,
    max_visible_objects: int,
    ingestion_state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return first-seen static object-parent facts from normal simulator state.

    Model prompts are built from the observation and memory protocol fields.
    The subject must be visible, but its non-pickupable support need not be
    separately listed as visible (AI2-THOR can expose an attached HandTowel while
    omitting its HandTowelHolder). Subjects that are ever picked up in the context
    session are excluded because their initial metadata parent may become stale;
    their locations are already recorded from the model-visible PickUp/PutObject
    action feedback.
    """
    cache = ingestion_state.setdefault("static_relation_cache", {})
    session_cache = cache.get(session_index)
    if session_cache is None:
        metadata = load_scene_metadata(session.get("scene"))
        objects_by_id = {
            str(obj.get("objectId")): obj
            for obj in metadata.get("objects") or []
            if obj.get("objectId")
        }
        operated_ids = {
            str((item.get("action") or {}).get("target"))
            for item in context_steps(session)
            if (item.get("action") or {}).get("action_type")
            in {"PickUp", "PickupObject"}
            and (item.get("action") or {}).get("target")
        }
        session_cache = {
            "objects_by_id": objects_by_id,
            "operated_ids": operated_ids,
        }
        cache[session_index] = session_cache

    visible_ids = [
        str(obj.get("object_id") or obj.get("objectId"))
        for obj in (step.get("visible_objects") or [])[:max_visible_objects]
        if obj.get("object_id") or obj.get("objectId")
    ]
    emitted = ingestion_state.setdefault("emitted_static_relations", set())
    updates: list[dict[str, Any]] = []
    for object_id in visible_ids:
        object_meta = session_cache["objects_by_id"].get(object_id)
        if (
            not object_meta
            or not object_meta.get("pickupable")
            or object_id in session_cache["operated_ids"]
        ):
            continue
        parent_id = _non_floor_parent(object_meta)
        if not parent_id:
            continue
        relation_key = (session_index, object_id, parent_id)
        if relation_key in emitted:
            continue
        subject_label = _object_label(object_id, object_labels, context_labels)
        target_label = _object_label(parent_id, object_labels, context_labels)
        if not subject_label or not target_label:
            continue
        parent_meta = session_cache["objects_by_id"].get(parent_id) or {}
        updates.append(
            {
                "subject_label": subject_label,
                "subject_type": object_meta.get("objectType") or object_type(object_id),
                "relation": "on_or_in",
                "target_label": target_label,
                "target_type": parent_meta.get("objectType") or object_type(parent_id),
            }
        )
        emitted.add(relation_key)
    return updates


def compact_context_observation(
    *,
    session_index: int,
    step_index: int,
    session: dict[str, Any],
    step: dict[str, Any],
    object_labels: dict[str, str] | None,
    context_labels: dict[str, str] | None,
    label_items: list[tuple[str, str]],
    max_visible_objects: int,
    ingestion_state: dict[str, Any] | None = None,
    context_ingestion_mode: str = "legacy",
) -> dict[str, Any]:
    if context_ingestion_mode not in {"legacy", "aligned"}:
        raise ValueError(f"Unknown context_ingestion_mode: {context_ingestion_mode}")
    raw_name = session.get("session_name") or f"context_session_{session_index}"
    session_name = strip_object_ids(str(raw_name))
    room = context_room_name(session)
    action_text, success, feedback_text = _action_text(
        step,
        object_labels,
        context_labels,
        label_items,
        preserve_durable_failure=context_ingestion_mode == "aligned",
    )
    action = step.get("action") or {}
    target_label = _object_label(action.get("target"), object_labels, context_labels)
    visible_step = step
    if context_ingestion_mode == "aligned":
        visible_step = dict(step)
        visible_step["visible_objects"] = _rotate_context_visible_objects(
            session_index=session_index,
            step=step,
            max_visible_objects=max_visible_objects,
            ingestion_state=ingestion_state if ingestion_state is not None else {},
        )
    observation = {
        "stage": "context_ingestion",
        "session_index": session_index,
        "step_index": step_index,
        "session_name": session_name,
        "room": room,
        "visible_objects": _visible_object_names(
            visible_step,
            object_labels,
            context_labels,
            limit=max_visible_objects,
        ),
        "action": {
            "action_type": action.get("action_type") or action.get("action"),
            "target_label": target_label or None,
            "text": action_text,
        },
        "feedback": {
            "success": success,
            "message": feedback_text,
        },
    }
    if context_ingestion_mode == "aligned":
        observation["held_object"] = (ingestion_state or {}).get("held_object")
        observation["new_visible_relations"] = static_visible_relation_updates(
            session_index=session_index,
            session=session,
            step=visible_step,
            object_labels=object_labels,
            context_labels=context_labels,
            max_visible_objects=max_visible_objects,
            ingestion_state=ingestion_state if ingestion_state is not None else {},
        )
    return observation


def advance_context_ingestion_state(
    *,
    observation: dict[str, Any],
    ingestion_state: dict[str, Any],
) -> dict[str, Any]:
    """Advance exact held-object state from one model-visible context step."""
    action = observation.get("action") or {}
    feedback = observation.get("feedback") or {}
    if not bool(feedback.get("success")):
        return ingestion_state
    action_type = action.get("action_type")
    target_label = action.get("target_label")
    if target_label and action_type == "PickUp":
        ingestion_state["held_object"] = target_label
    elif action_type == "PutObject":
        ingestion_state["held_object"] = None
    return ingestion_state


def action_description(action: Any) -> str:
    if not isinstance(action, dict):
        return str(action)
    if action.get("natural_language"):
        return str(action["natural_language"])
    action_type = action.get("action_type") or action.get("action") or "Action"
    target = (
        action.get("target_label") or action.get("target_type") or action.get("target")
    )
    if target:
        return f"{action_type} {target}"
    return str(action_type)


def observe_memorymodule_step(
    memory_runtime: MemoryRuntime, info: dict[str, Any]
) -> str | None:
    try:
        memory_runtime.observe_step(info)
        return None
    except Exception as exc:
        return short_error(exc)


def execute_action_sequence(
    *,
    env: EmbodiedMemorizerEnv,
    obs: dict[str, Any],
    payload: dict[str, Any],
    raw_text: str,
    memory_runtime: MemoryRuntime,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], bool, list[dict[str, Any]], str]:
    executed = []
    done = False
    stopped_reason = "sequence_exhausted"
    for sequence_index, planned_action in enumerate(
        action_sequence_from_payload(payload)
    ):
        action = planned_action
        sequence_binding = None
        grounding_correction = None
        observation_binding = None
        action, sequence_binding = bind_sequence_action_to_observation_space(
            action, obs
        )
        raw_payload = {
            "sequence_index": sequence_index,
            "planned_action": planned_action,
        }
        action, grounding_correction = repair_action_id_from_structured_prediction(
            action,
            raw_payload,
            obs,
        )
        action, observation_binding = bind_action_to_observation_space(action, obs)
        if sequence_binding:
            raw_payload["sequence_action_binding"] = sequence_binding
        if grounding_correction:
            raw_payload["action_id_grounding_correction"] = grounding_correction
        if observation_binding:
            raw_payload["action_id_observation_binding"] = observation_binding

        if action == -1:
            current_result = env.current_result()
            feedback = (
                "The planned future action is no longer available after the "
                "previous state transition and memory action filtering. Replan "
                "from the current observation."
            )
            executed.append(
                {
                    "sequence_index": sequence_index,
                    "planned_action": planned_action,
                    "executed_action": None,
                    "success": False,
                    "reward": 0.0,
                    "feedback": feedback,
                    "task_success": int(current_result.task_completed)
                    if current_result
                    else 0,
                    "task_progress": current_result.task_progress
                    if current_result
                    else 0.0,
                    "env_step": env._current_step,
                    "binding": raw_payload,
                    "not_executed": True,
                }
            )
            stopped_reason = "sequence_action_unavailable_after_refilter"
            break

        try:
            obs, reward, done, info = env.step(
                action,
                reasoning=json.dumps(
                    payload.get("reasoning") or {}, ensure_ascii=False
                ),
            )
        except Exception as exc:
            error_text = short_error(exc)
            runtime_info = {
                "action_description": action_description(action),
                "env_feedback": f"Simulator step error: {error_text}",
                "last_action_success": 0.0,
                "action_entry": action if isinstance(action, dict) else {},
                "task_progress": 0.0,
            }
            step_record = {
                "sequence_index": sequence_index,
                "planned_action": planned_action,
                "executed_action": action,
                "success": False,
                "reward": -1.0,
                "feedback": f"Simulator step error: {error_text}",
                "binding": raw_payload,
            }
            if args.auto_memory_observe_step:
                memory_update_error = observe_memorymodule_step(
                    memory_runtime, runtime_info
                )
                if memory_update_error:
                    step_record["memory_update_error"] = memory_update_error
            executed.append(step_record)
            stopped_reason = "simulator_step_error"
            return obs, True, executed, stopped_reason

        step_record = {
            "sequence_index": sequence_index,
            "planned_action": planned_action,
            "executed_action": action,
            "success": bool(info.get("last_action_success", 0.0)),
            "reward": float(reward),
            "feedback": info.get("env_feedback"),
            "task_success": info.get("task_success", 0),
            "task_progress": info.get("task_progress", 0.0),
            "env_step": info.get("env_step"),
            "binding": raw_payload,
        }
        runtime_info = dict(info)
        runtime_info.setdefault("action_description", action_description(action))
        runtime_info.setdefault("env_feedback", info.get("env_feedback"))
        runtime_info.setdefault(
            "last_action_success", info.get("last_action_success", 0.0)
        )
        runtime_info.setdefault(
            "action_entry", action if isinstance(action, dict) else {}
        )
        runtime_info.setdefault("action", action if isinstance(action, dict) else {})
        if args.auto_memory_observe_step:
            memory_update_error = observe_memorymodule_step(
                memory_runtime, runtime_info
            )
            if memory_update_error:
                step_record["memory_update_error"] = memory_update_error
        executed.append(step_record)
        if not done:
            obs = memory_runtime.filter_observation(obs)
        if done:
            stopped_reason = "task_done_or_max_steps"
            break
        if not step_record["success"]:
            stopped_reason = "action_failed"
            break
        if env._current_step >= args.max_steps:
            done = True
            stopped_reason = "max_steps"
            break
    return obs, done, executed, stopped_reason
