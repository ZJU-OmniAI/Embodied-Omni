"""规则类型判断、场景池选取、显示名称和模板渲染工具。"""

from ...experience_templates import ConstraintRule, ExperienceType

# AI2-THOR object type -> English display name
_DISPLAY_NAME_MAP = {
    "Fork": "Fork",
    "Spoon": "Spoon",
    "Knife": "Knife",
    "PotatoChip": "PotatoChip",
    "Bowl": "Bowl",
    "Plate": "Plate",
    "Mug": "Mug",
    "Cup": "Cup",
    "Microwave": "Microwave",
    "Toaster": "Toaster",
    "CoffeeMachine": "CoffeeMachine",
    "SpaceHeater": "SpaceHeater",
    "LightSwitch": "LightSwitch",
    "Drawer": "Drawer",
    "Cabinet": "Cabinet",
}


# ── 规则类型判断 ─────────────────────────────────────────────


def is_owner_habit_rule(rule: ConstraintRule) -> bool:
    return (
        getattr(rule, "rule_family", ExperienceType.PHYSICAL_CONSTRAINT)
        == ExperienceType.OWNER_HABIT
    )


def auto_failure_message(rule: ConstraintRule) -> str:
    if is_owner_habit_rule(rule):
        return (
            "Preference Violation: this placement does not match the household habit."
        )
    return "Action Failed: this interaction violates a known household constraint."


def room_pool_for_rule(
    rule: ConstraintRule, scene_pool: list[str] | None = None
) -> list[str]:
    if scene_pool:
        return list(scene_pool)
    if is_owner_habit_rule(rule):
        return (
            [f"FloorPlan{i}" for i in range(1, 31)]
            + [f"FloorPlan{200 + i}" for i in range(1, 31)]
            + [f"FloorPlan{300 + i}" for i in range(1, 31)]
            + [f"FloorPlan{400 + i}" for i in range(1, 31)]
        )
    return [f"FloorPlan{i}" for i in range(1, 31)]


# ── 显示 / 模板工具 ──────────────────────────────────────────


def get_object_display_name(obj: dict) -> str:
    return _DISPLAY_NAME_MAP.get(obj.get("objectType", ""), obj.get("objectType", ""))


def render_nl_template(template: str, slots: dict[str, str]) -> str:
    if not template:
        return ""
    try:
        return template.format(**slots)
    except Exception:
        return template


def first_filter_value(value: str) -> str:
    tokens = [t.strip() for t in str(value).split("|") if t.strip()]
    return tokens[0] if tokens else str(value)


def sequence_uses_slot(seq: list[dict], slot: str) -> bool:
    return any(
        step.get("target_slot") == slot or step.get("instrument_slot") == slot
        for step in seq
        if isinstance(step, dict)
    )
