"""
Embodied Memorizer Bench - 数据 Schema 定义

定义 Episode、Session、Trajectory Step、Probe 等核心数据结构。
所有数据最终序列化为 JSON 格式，图像文件单独存储。
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum
import json


# ─── 枚举类型 ──────────────────────────────────────────────────


class DifficultyLevel(str, Enum):
    L0 = "L0"  # 反应式任务
    L1 = "L1"  # 工作记忆任务
    L2 = "L2"  # 检索依赖任务
    L3 = "L3"  # 经验泛化任务


class MemoryCueType(str, Enum):
    PASSIVE_OBSERVATION = "passive_observation"  # 被动观察到某物体
    FAILURE_EVENT = "failure_event"  # 经历失败交互
    STATE_CHANGE = "state_change"  # 物体状态变化
    SELF_ACTION = "self_action"  # 自身操作改变了物体位置
    IMPLICIT_RULE = "implicit_rule"  # 多次相似经历构成可归纳规律


class ActionType(str, Enum):
    """AI2-THOR 兼容的动作类型"""

    NAVIGATE = "Navigate"
    PICKUP = "PickUp"
    PUT = "PutObject"
    OPEN = "Open"
    CLOSE = "Close"
    TOGGLE_ON = "ToggleOn"
    TOGGLE_OFF = "ToggleOff"
    SLICE = "Slice"
    TRANSFER_CONTENTS = "TransferContents"
    ROTATE_LEFT = "RotateLeft"
    ROTATE_RIGHT = "RotateRight"
    LOOK_UP = "LookUp"
    LOOK_DOWN = "LookDown"
    MOVE_FORWARD = "MoveForward"
    # 自定义动作（AI2-THOR 不直接支持，在标注层模拟）
    PLUGIN = "PlugIn"
    UNPLUG = "Unplug"
    END = "End"
    # 房间级导航（Session 间过渡）
    LEAVE_ROOM = "LeaveRoom"
    ENTER_ROOM = "EnterRoom"


class PenaltyType(str, Enum):
    BLIND_NAVIGATION = "blind_navigation"  # 盲目导航到错误位置
    STATE_REVERSAL = "state_reversal"  # 短时间内执行抵消性动作
    STALE_SPATIAL_MEMORY = "stale_spatial_memory"  # 凭僵化记忆去过时的位置
    CRITICAL_ERR = "critical_err"  # 重复已知致命错误


# ─── 基础数据结构 ──────────────────────────────────────────────


@dataclass
class Position:
    x: float
    y: float
    z: float


@dataclass
class AgentState:
    position: Position
    rotation_y: float  # 水平朝向角度 (0-360)
    camera_horizon: float = 0.0  # 相机俯仰角
    holding: Optional[str] = None  # 手持物体的 object_id


@dataclass
class ObjectState:
    """场景中物体的状态快照"""

    object_id: str  # AI2-THOR 格式: "Type|+x|+y|+z"
    object_type: str  # 如 "Apple", "Drawer"
    position: Position
    parent_receptacle: Optional[str] = None
    # 关键属性
    pickupable: bool = False
    openable: bool = False
    toggleable: bool = False
    receptacle: bool = False
    is_open: bool = False
    is_toggled: bool = False
    is_locked: bool = False  # 自定义属性（AI2-THOR 无原生支持）
    temperature: str = "RoomTemp"  # "RoomTemp" / "Hot" / "Cold"
    is_dirty: bool = False
    is_broken: bool = False
    salient_materials: list[str] = field(default_factory=list)  # ["Metal"], ["Wood"] 等


@dataclass
class VisibleObject:
    """观察中可见的物体（简化版）"""

    object_id: str
    object_type: str
    position: Position
    distance: float  # 与智能体的距离
    in_view_center: bool = False  # 是否在视野中心（vs 边缘）


# ─── 隐式规则 ──────────────────────────────────────────────────


@dataclass
class HiddenRule:
    """Episode 中的隐式规则，模型需要通过经验归纳发现"""

    rule_id: str
    description: str
    rule_type: str  # "interaction_constraint", "state_propagation", "spatial_pattern"
    condition: dict  # 触发条件
    effect: str  # 效果描述（如 "Action Failed: Power Trip"）
    related_sessions: list[int] = field(default_factory=list)  # 在哪些 Session 中体现


# ─── 记忆线索 ──────────────────────────────────────────────────


@dataclass
class MemoryCue:
    """需要模型记住的信息片段"""

    cue_id: str
    cue_type: MemoryCueType
    description: str
    planted_in_session: int  # 在哪个 Session 植入
    planted_at_step: int  # 在哪一步植入
    object_id: Optional[str] = None  # 关联的物体
    initial_location: Optional[str] = None
    tested_in_probe: Optional[str] = None  # 在哪个 Probe 中测试


# ─── 轨迹步骤 ──────────────────────────────────────────────────


@dataclass
class ActionRecord:
    """单个动作记录"""

    action_type: str  # ActionType value
    target: Optional[str] = None  # 目标物体 ID
    instrument: Optional[str] = None  # 使用的工具/物体
    natural_language: str = ""  # 自然语言描述


@dataclass
class FeedbackRecord:
    """动作执行反馈"""

    success: bool
    message: str = ""
    event_memory_tag: Optional[str] = None  # 标记为需要记忆的事件


@dataclass
class MemoryCueExposure:
    """记忆线索的曝光记录"""

    cue_id: str
    exposure_type: str  # "peripheral_vision", "direct_interaction", "failure_feedback"
    description: str


@dataclass
class TrajectoryStep:
    """轨迹中的单步记录"""

    step_id: int
    # 观察
    image_path: str
    visible_objects: list[VisibleObject]
    agent_state: AgentState
    # 动作
    action: ActionRecord
    # 反馈
    feedback: FeedbackRecord
    # 记忆线索曝光（如果有）
    memory_cue_exposed: Optional[MemoryCueExposure] = None


# ─── 评测钩子 ──────────────────────────────────────────────────


@dataclass
class EvaluationTrigger:
    """单个评测触发器"""

    trigger_condition: str  # 触发条件描述
    penalty_type: PenaltyType
    weight: float = 1.0  # 惩罚权重
    description: str = ""


@dataclass
class EvaluationHooks:
    """Probe 的评测逻辑"""

    success_condition: str  # 成功判定条件
    rar_triggers: list[EvaluationTrigger] = field(default_factory=list)
    err_triggers: list[EvaluationTrigger] = field(default_factory=list)


@dataclass
class ExpectedAction:
    """Probe 的预期动作"""

    action_type: str
    target: Optional[str] = None
    object_held: Optional[str] = None


# ─── Probe 定义 ──────────────────────────────────────────────


@dataclass
class StateChange:
    """Probe 执行后引起的状态变化"""

    object_id: str
    change_type: str  # "position_change", "state_change"
    old_value: str
    new_value: str
    description: str


@dataclass
class MicroProbe:
    """Session 结束后的短程记忆探针"""

    probe_id: str
    probe_level: DifficultyLevel
    instruction: str  # 自然语言指令
    context: str  # 背景说明
    scene: str  # 执行 Probe 的场景
    # 预期动作序列（ground truth）
    expected_actions: list[ExpectedAction]
    optimal_steps: int
    max_steps: int = 30
    # 评测
    evaluation_hooks: EvaluationHooks = field(
        default_factory=lambda: EvaluationHooks(success_condition="")
    )
    # Probe 引起的状态变化（影响后续 Session/Probe）
    state_changes: list[StateChange] = field(default_factory=list)


@dataclass
class MemoryRequirement:
    """Macro-Probe 需要整合的记忆"""

    memory_type: (
        str  # "event_generalization", "dynamic_spatial_tracking", "episodic_retrieval"
    )
    source_sessions: list[int]
    source_events: list[str]
    expected_inference: str


@dataclass
class SubTask:
    """Macro-Probe 中的子任务"""

    sub_task_id: str
    instruction: str
    expected_actions: list[ExpectedAction]
    evaluation_hooks: EvaluationHooks


@dataclass
class MacroProbe:
    """Episode 末尾的终极泛化探针"""

    probe_id: str
    probe_level: DifficultyLevel
    instruction: str
    scene: str
    required_memory_integration: list[MemoryRequirement]
    sub_tasks: list[SubTask]
    max_steps: int = 60
    scoring: dict = field(
        default_factory=lambda: {"lambda_rar": 5, "alpha": 1.0, "beta": 2.0}
    )


# ─── Session ──────────────────────────────────────────────────


@dataclass
class ContextTrajectory:
    """Context Session 中的专家轨迹"""

    description: str
    total_steps: int
    steps: list[TrajectoryStep]


@dataclass
class Session:
    """一个 Session 包含 Context 轨迹 + 可选的 Micro-Probe"""

    session_id: str
    session_name: str
    scene: str  # AI2-THOR 场景名 (如 "FloorPlan1")
    room_type: str  # "Kitchen" / "LivingRoom" / "Bedroom" / "Bathroom"
    context_trajectory: ContextTrajectory
    micro_probe: Optional[MicroProbe] = None


# ─── Episode (顶层) ──────────────────────────────────────────


@dataclass
class Episode:
    """完整的评测 Episode"""

    episode_id: str
    episode_name: str
    description: str
    difficulty: DifficultyLevel
    # 隐式规则和记忆线索
    hidden_rules: list[HiddenRule] = field(default_factory=list)
    memory_cues: list[MemoryCue] = field(default_factory=list)
    # Session 序列
    sessions: list[Session] = field(default_factory=list)
    # 终极 Probe
    macro_probe: Optional[MacroProbe] = None
    # 场景来源元信息（记录该 Episode 中各 Session 使用的场景）
    scene_metadata: dict = field(default_factory=dict)


# ─── 序列化工具 ──────────────────────────────────────────────


def episode_to_json(episode: Episode, indent: int = 2) -> str:
    """将 Episode 序列化为 JSON 字符串"""

    def convert(obj):
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items() if v is not None}
        return obj

    return json.dumps(convert(episode), indent=indent, ensure_ascii=False)


def save_episode(episode: Episode, path: str):
    """保存 Episode 到 JSON 文件"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(episode_to_json(episode))


def load_episode(path: str) -> dict:
    """加载 Episode JSON（返回 dict，不重建 dataclass）"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
