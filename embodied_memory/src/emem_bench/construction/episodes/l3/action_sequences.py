"""
失败动作序列与 Probe 解答动作序列的生成。

负责两类"动作蓝图"的构造：
  1. failure sequence  - Context Session 中触发失败的动作链
  2. probe sequence    - MacroProbe 的正确解答动作链
"""

from typing import Optional

from ...schema import ExpectedAction
from ...experience_templates import ConstraintRule
from .rule_helpers import (
    is_owner_habit_rule,
    auto_failure_message,
    get_object_display_name,
    render_nl_template,
    first_filter_value,
)
from .object_finders import find_safe_container

_ENCLOSED_APPLIANCES = {"Microwave", "Fridge"}
_OPEN_TOP_APPLIANCES = {"StoveBurner", "Toaster", "CoffeeMachine"}


# ═══════════════════════════════════════════════════════════════
# Probe 解答序列（正确行为蓝图）
# ═══════════════════════════════════════════════════════════════


def probe_solution_sequence(rule: ConstraintRule) -> list[dict]:
    """返回 MacroProbe 的正确动作模板序列（含 slot 占位符）。"""
    if is_owner_habit_rule(rule):
        return [
            {
                "action": "Navigate",
                "target_slot": "task_object",
                "nl_template": "Navigate to {task_nl}",
            },
            {
                "action": "PickUp",
                "target_slot": "task_object",
                "nl_template": "Pick up {task_nl}",
            },
            {
                "action": "Navigate",
                "target_slot": "preferred_target",
                "nl_template": "Navigate to the homeowner-preferred storage location",
            },
            {
                "action": "PutObject",
                "target_slot": "preferred_target",
                "nl_template": "Place {task_nl} in the homeowner-preferred storage location",
            },
        ]

    target_type = first_filter_value(rule.target_filter.value)
    if target_type in _ENCLOSED_APPLIANCES:
        return [
            {
                "action": "Navigate",
                "target_slot": "task_object",
                "nl_template": "Navigate to {task_nl}",
            },
            {
                "action": "PickUp",
                "target_slot": "task_object",
                "nl_template": "Pick up {task_nl}",
            },
            {
                "action": "Navigate",
                "target_slot": "aux_object",
                "nl_template": "Navigate to safe container {aux_nl}",
            },
            {
                "action": "TransferContents",
                "target_slot": "aux_object",
                "instrument_slot": "task_object",
                "nl_template": "Transfer the contents of {task_nl} into {aux_nl}",
            },
            {
                "action": "PickUp",
                "target_slot": "aux_object",
                "nl_template": "Pick up {aux_nl} with the transferred contents",
            },
            {
                "action": "Navigate",
                "target_slot": "target_object",
                "nl_template": "Navigate to {target_nl}",
            },
            {
                "action": "Open",
                "target_slot": "target_object",
                "nl_template": "Open {target_nl}",
            },
            {
                "action": "PutObject",
                "target_slot": "target_object",
                "nl_template": "Put the safe container into {target_nl}",
            },
            {
                "action": "Close",
                "target_slot": "target_object",
                "nl_template": "Close {target_nl}",
            },
            {
                "action": "ToggleOn",
                "target_slot": "target_object",
                "nl_template": "Turn on {target_nl} to complete the processing",
            },
        ]
    if target_type in _OPEN_TOP_APPLIANCES:
        return [
            {
                "action": "Navigate",
                "target_slot": "task_object",
                "nl_template": "Navigate to {task_nl}",
            },
            {
                "action": "PickUp",
                "target_slot": "task_object",
                "nl_template": "Pick up {task_nl}",
            },
            {
                "action": "Navigate",
                "target_slot": "aux_object",
                "nl_template": "Navigate to safe container {aux_nl}",
            },
            {
                "action": "TransferContents",
                "target_slot": "aux_object",
                "instrument_slot": "task_object",
                "nl_template": "Transfer the contents of {task_nl} into {aux_nl}",
            },
            {
                "action": "PickUp",
                "target_slot": "aux_object",
                "nl_template": "Pick up {aux_nl} with the transferred contents",
            },
            {
                "action": "Navigate",
                "target_slot": "target_object",
                "nl_template": "Navigate to {target_nl}",
            },
            {
                "action": "PutObject",
                "target_slot": "target_object",
                "nl_template": "Put the safe container on {target_nl}",
            },
            {
                "action": "ToggleOn",
                "target_slot": "target_object",
                "nl_template": "Turn on {target_nl} to complete the processing",
            },
        ]
    return [
        {
            "action": "Navigate",
            "target_slot": "task_object",
            "nl_template": "Navigate to {task_nl}",
        },
        {
            "action": "PickUp",
            "target_slot": "task_object",
            "nl_template": "Pick up {task_nl}",
        },
    ]


def materialize_probe_solution_actions(
    rule: ConstraintRule,
    task_object: Optional[dict] = None,
    target_object: Optional[dict] = None,
    aux_object: Optional[dict] = None,
    preferred_object: Optional[dict] = None,
) -> list[ExpectedAction]:
    """将 probe_solution_sequence 的模板序列实例化为 ExpectedAction 列表。"""
    slot_targets = {
        "task_object": task_object.get("objectId") if task_object else None,
        "failure_object": task_object.get("objectId") if task_object else None,
        "target_object": target_object.get("objectId") if target_object else None,
        "aux_object": aux_object.get("objectId") if aux_object else None,
        "safe_object": aux_object.get("objectId") if aux_object else None,
        "preferred_target": preferred_object.get("objectId")
        if preferred_object
        else None,
    }
    slot_objects = {
        "task_object": task_object,
        "failure_object": task_object,
        "target_object": target_object,
        "aux_object": aux_object,
        "safe_object": aux_object,
        "preferred_target": preferred_object,
    }

    expected_actions: list[ExpectedAction] = []
    for raw in probe_solution_sequence(rule):
        action_type = str(raw.get("action", "")).strip()
        if not action_type:
            continue
        target_slot = raw.get("target_slot")
        instrument_slot = raw.get("instrument_slot")
        target = slot_targets.get(target_slot, raw.get("target"))
        object_held = slot_targets.get(instrument_slot) if instrument_slot else None
        if target_slot and target is None:
            continue
        target_obj = slot_objects.get(target_slot) if target_slot else None
        if (
            action_type == "PutObject"
            and target
            and target_obj
            and target_obj.get("openable")
            and (
                not expected_actions
                or expected_actions[-1].action_type != "Open"
                or expected_actions[-1].target != target
            )
        ):
            expected_actions.append(
                ExpectedAction(
                    action_type="Open",
                    target=target,
                )
            )
        expected_actions.append(
            ExpectedAction(
                action_type=action_type,
                target=target,
                object_held=object_held,
            )
        )
    return expected_actions


def find_auxiliary_object(
    meta: dict,
    rule: ConstraintRule,
    avoid_object_id: str = "",
) -> Optional[dict]:
    """返回 Probe 中用作安全替代容器的物体（owner_habit 规则不需要）。"""
    if is_owner_habit_rule(rule):
        return None
    obj_type = first_filter_value(rule.object_filter.value)
    return find_safe_container(
        meta, failure_object_type=obj_type, avoid_object_id=avoid_object_id
    )


def macro_instruction(
    rule: ConstraintRule,
    task_object: Optional[dict] = None,
    target_object: Optional[dict] = None,
) -> str:
    obj_type = first_filter_value(rule.object_filter.value)
    task_nl = get_object_display_name(task_object) if task_object else obj_type
    tgt_value = rule.target_filter.value if rule.target_filter else ""
    target_nl = (
        get_object_display_name(target_object)
        if target_object
        else first_filter_value(tgt_value)
    )
    if is_owner_habit_rule(rule):
        return f"Please store this {task_nl} in the location that best matches this household's implicit habit."
    if target_object and target_object.get("objectType") == "Microwave":
        return f"Please safely heat the contents of this {task_nl} without triggering the safety alarm on {target_nl}."
    return f"Please safely handle this {task_nl}, complete the task involving {target_nl}, and avoid the known failure."


# ═══════════════════════════════════════════════════════════════
# 失败动作序列（Context Session 中触发失败事件）
# ═══════════════════════════════════════════════════════════════


def failure_action_sequence(
    rule: ConstraintRule,
    session_idx: int,
    obj_nl: str,
    tgt_nl: str,
) -> list[dict]:
    """返回触发失败事件的动作模板序列（含 slot 占位符）。"""
    if is_owner_habit_rule(rule):
        return [
            {
                "action": "Navigate",
                "target_slot": "failure_object",
                "nl_template": f"Navigate to {obj_nl}",
            },
            {
                "action": "PickUp",
                "target_slot": "failure_object",
                "nl_template": f"Pick up {obj_nl}",
            },
            {
                "action": "Navigate",
                "target_slot": "target_object",
                "nl_template": f"Navigate to {tgt_nl}",
            },
            {
                "action": "PutObject",
                "target_slot": "target_object",
                "nl_template": f"Try placing {obj_nl} casually on {tgt_nl}",
                "simulate_failure": True,
                "event_memory_tag": f"failure_{session_idx}",
            },
        ]

    target_type = rule.target_filter.value if rule.target_filter else ""
    if target_type in _ENCLOSED_APPLIANCES:
        return [
            {
                "action": "Navigate",
                "target_slot": "failure_object",
                "nl_template": f"Navigate to {obj_nl} to begin the task",
            },
            {
                "action": "PickUp",
                "target_slot": "failure_object",
                "nl_template": f"Pick up {obj_nl}",
            },
            {
                "action": "Navigate",
                "target_slot": "target_object",
                "nl_template": f"Navigate to {tgt_nl}",
            },
            {
                "action": "Open",
                "target_slot": "target_object",
                "nl_template": f"Open {tgt_nl}",
            },
            {
                "action": "PutObject",
                "target_slot": "target_object",
                "nl_template": f"Try putting {obj_nl} into {tgt_nl}",
                "simulate_failure": True,
                "event_memory_tag": f"failure_{session_idx}",
            },
            {
                "action": "Close",
                "target_slot": "target_object",
                "nl_template": f"Close {tgt_nl}",
            },
        ]
    if target_type in _OPEN_TOP_APPLIANCES:
        return [
            {
                "action": "Navigate",
                "target_slot": "failure_object",
                "nl_template": f"Navigate to {obj_nl}",
            },
            {
                "action": "PickUp",
                "target_slot": "failure_object",
                "nl_template": f"Pick up {obj_nl}",
            },
            {
                "action": "Navigate",
                "target_slot": "target_object",
                "nl_template": f"Navigate to {tgt_nl}",
            },
            {
                "action": "PutObject",
                "target_slot": "target_object",
                "nl_template": f"Try placing {obj_nl} on {tgt_nl}",
                "simulate_failure": True,
                "event_memory_tag": f"failure_{session_idx}",
            },
        ]
    # 默认：电路类规则
    return [
        {
            "action": "Navigate",
            "target_slot": "failure_object",
            "nl_template": f"Navigate to {obj_nl}",
        },
        {
            "action": "PickUp",
            "target_slot": "failure_object",
            "nl_template": f"Pick up {obj_nl}",
        },
        {
            "action": "Navigate",
            "target_slot": "target_object",
            "nl_template": f"Navigate to the outlet near {tgt_nl}",
        },
        {
            "action": "ToggleOn",
            "target_slot": "failure_object",
            "nl_template": f"Try plugging {obj_nl} into {tgt_nl}",
            "simulate_failure": True,
            "event_memory_tag": f"failure_{session_idx}",
        },
    ]


def materialize_correction_actions(
    rule: ConstraintRule,
    failure_object: dict,
    safe_object: Optional[dict] = None,
    preferred_object: Optional[dict] = None,
) -> list[dict]:
    """失败后的矫正动作序列（模拟人类干预，展示正确行为）。

    owner_habit: 将物体放回屋主习惯的正确位置（preferred_object）。
    physical_constraint: 记录拒绝执行的人工干预步骤，物体放回安全位置（safe_object）。
    """
    obj_nl = get_object_display_name(failure_object)

    if is_owner_habit_rule(rule) and preferred_object:
        preferred_nl = get_object_display_name(preferred_object)
        pref_id = preferred_object["objectId"]
        is_openable = preferred_object.get("openable", False)
        place_prep = "into" if is_openable else "on"
        actions = [
            {
                "action": "HumanIntervention",
                "nl": "Human intervention: the failed placement was corrected.",
            },
            {
                "action": "Navigate",
                "target": pref_id,
                "nl": f"Navigate to {preferred_nl}",
            },
        ]
        if is_openable:
            actions.append(
                {"action": "Open", "target": pref_id, "nl": f"Open {preferred_nl}"}
            )
        actions.append(
            {
                "action": "PutObject",
                "target": pref_id,
                "nl": f"Place {obj_nl} {place_prep} {preferred_nl}",
            }
        )
        if is_openable:
            actions.append(
                {"action": "Close", "target": pref_id, "nl": f"Close {preferred_nl}"}
            )
        return actions

    # physical_constraint: 记录人工干预拒绝，然后将物体放回安全位置
    actions = [
        {
            "action": "HumanIntervention",
            "nl": f"Human intervention: task rejected - {rule.incompatibility_rule}",
        },
    ]
    if safe_object:
        safe_nl = get_object_display_name(safe_object)
        actions.append(
            {
                "action": "Navigate",
                "target": safe_object["objectId"],
                "nl": f"Navigate to safe location {safe_nl} to return {obj_nl}",
            }
        )
        actions.append(
            {
                "action": "PutObject",
                "target": safe_object["objectId"],
                "nl": f"Place {obj_nl} on {safe_nl} (restored)",
            }
        )
    return actions


def materialize_failure_action_sequence(
    rule: ConstraintRule,
    failure_object: dict,
    target_object: dict,
    session_idx: int,
    safe_object: Optional[dict] = None,
    preferred_object: Optional[dict] = None,
) -> list[dict]:
    """将 failure_action_sequence 的模板序列实例化为含具体 objectId 的动作列表。"""
    obj_nl = get_object_display_name(failure_object)
    tgt_nl = get_object_display_name(target_object)
    safe_nl = get_object_display_name(safe_object) if safe_object else ""
    preferred_nl = get_object_display_name(preferred_object) if preferred_object else ""

    slot_targets = {
        "failure_object": failure_object.get("objectId"),
        "target_object": target_object.get("objectId"),
        "safe_object": safe_object.get("objectId") if safe_object else None,
        "preferred_target": preferred_object.get("objectId")
        if preferred_object
        else None,
    }
    slot_labels = {
        "obj_nl": obj_nl,
        "tgt_nl": tgt_nl,
        "safe_nl": safe_nl,
        "preferred_nl": preferred_nl,
    }

    materialized = []
    for raw in failure_action_sequence(rule, session_idx, obj_nl, tgt_nl):
        step = {
            "action": raw["action"],
            "target": slot_targets.get(raw.get("target_slot"), raw.get("target")),
            "instrument": slot_targets.get(
                raw.get("instrument_slot"), raw.get("instrument")
            ),
            "nl": render_nl_template(
                raw.get("nl_template") or raw.get("nl", ""), slot_labels
            ),
            "simulate_failure": raw.get("simulate_failure", False),
        }
        if step["simulate_failure"]:
            step["failure_message"] = auto_failure_message(rule)
            step["event_memory_tag"] = raw.get(
                "event_memory_tag", f"failure_{session_idx}"
            )
        materialized.append(step)
    return materialized
