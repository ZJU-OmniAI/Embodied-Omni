"""L0/L1 任务的基础数据结构"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class ActionStep:
    """单步原子动作"""

    action: str  # Navigate, PickUp, PutObject, Open, Close, ToggleOn, ToggleOff
    target: str  # objectId
    target_type: str  # objectType
    nl: str  # 自然语言描述
    simulate_failure: bool = False
    failure_message: str = ""
    event_memory_tag: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"action": self.action, "target": self.target, "nl": self.nl}
        if self.simulate_failure:
            d["simulate_failure"] = True
            d["failure_message"] = self.failure_message
        if self.event_memory_tag:
            d["event_memory_tag"] = self.event_memory_tag
        return d


@dataclass
class Task:
    """L0/L1 单个任务，由一组 ActionStep 构成"""

    name: str  # 自然语言任务名
    level: str  # "L0" 或 "L1"
    task_type: str  # pick_and_place, clean_and_place, heat_and_place, ...
    steps: list[ActionStep]
    key_object: str = ""  # 核心操作的物体 ID（用于去重采样）

    @property
    def step_count(self) -> int:
        return len(self.steps)
