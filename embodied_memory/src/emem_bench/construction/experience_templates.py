"""
经验模板定义

ConstraintRule 只保留 constraint_discovery 输出的核心字段。
规则加载由调用方通过 ConstraintDiscoverer.load_discovered_dir() 完成，
不再维护全局注册表。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class ExperienceType(str):
    PHYSICAL_CONSTRAINT = "physical_constraint"
    OWNER_HABIT = "owner_habit"


@dataclass
class PropertyFilter:
    """从场景元数据中动态匹配物体的过滤器。"""

    property_key: str  # 元数据字段名，如 "salientMaterials" / "objectType"
    value: str  # 期望值，支持 "|" 分隔多值


@dataclass
class ConstraintRule:
    """一条约束规则的核心定义。"""

    rule_id: str
    rule_family: str  # "physical_constraint" | "owner_habit"
    object_filter: PropertyFilter  # 规则适用的物体
    incompatibility_rule: str  # 规则的自然语言描述

    # physical_constraint: 不兼容的目标设备
    # owner_habit: 可选（不需要负例）
    target_filter: Optional[PropertyFilter] = None

    # owner_habit: 正确的放置位置
    preferred_target_filter: Optional[PropertyFilter] = None

    # 来自哪些场景的 RelationSample 支撑了这条规则
    source_scenes: list[str] = field(default_factory=list)


@dataclass
class GeneralizationProbeTemplate:
    """暂时保留此类以兼容 L3 generator 的导入，字段待 L3 适配阶段补充。"""

    nl_instruction_template: str = ""
    generalization_object_type: str = ""
    generalization_object_filter: Optional[PropertyFilter] = None
    auxiliary_object_filter: Optional[PropertyFilter] = None
    solution_action_sequence: list = field(default_factory=list)
    forbidden_action_patterns: list = field(default_factory=list)
    err_trigger_description: str = ""


def get_probe_template(_: ConstraintRule) -> Optional[GeneralizationProbeTemplate]:
    """暂时返回 None，待 L3 generator 适配阶段实现。"""
    return None
