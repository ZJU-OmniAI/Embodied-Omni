"""
记忆工具定义，供 VLM function calling 使用。

设计模式参考 Mem-T：每个工具独立一个类，包含 schema 定义和执行逻辑，
由 MemoryToolRegistry 统一管理。

工具列表：RememberObjectTool、RememberEventTool、RememberSceneTool、
QueryExperienceTool、MemoryQueryTool
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional

from .memory import MemorySystem


# ── BaseTool ──────────────────────────────────────────────────────────────────


class BaseTool(ABC):
    """所有记忆工具的抽象基类。

    子类在 __init__ 中调用 super().__init__() 完成 schema 注册，
    并实现 __call__ 提供实际执行逻辑。

    Parameters
    ----------
    name : str
        工具名称，即 VLM 调用时使用的函数名，如 "remember_object"。
    description : str
        工具功能的自然语言描述，直接注入 system prompt，
        模型依据此描述决定何时调用该工具。
    parameters : Dict[str, Any]
        参数的 JSON Schema 定义，格式为 {参数名: schema_dict}。
        每个 schema_dict 至少包含 "type" 和 "description" 字段，
        可选 "enum" 限制取值范围。该字段对应 OpenAI function calling
        的 "properties" 字段，模型按此生成参数值。
    required : Optional[List[str]]
        必填参数名列表。默认为 parameters 的全部 key。
        未列入此列表的参数模型可以省略，调用时需提供默认值。
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        # required 未指定时默认所有参数均为必填
        self.required = required if required is not None else list(parameters.keys())

    def to_schema(self) -> Dict[str, Any]:
        """生成符合 OpenAI function calling 格式的 JSON Schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }

    @abstractmethod
    def __call__(self, memory: MemorySystem, **kwargs) -> Any:
        """执行工具逻辑，kwargs 对应模型传入的参数。"""
        pass


# ── 具体工具类 ─────────────────────────────────────────────────────────────────


class RememberObjectTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="remember_object",
            description=(
                "Record a spatial object into memory: its type, relationships to other objects, "
                "and observable properties. Call this whenever you observe or learn about an "
                "object's existence, location, or state."
            ),
            parameters={
                "name": {
                    "type": "string",
                    "description": "Object name, e.g. 'Apple', 'CounterTop', 'Fridge'.",
                },
                "node_type": {
                    "type": "string",
                    "description": "Category of the object.",
                    "enum": ["object", "receptacle", "room", "landmark"],
                },
                "relations": {
                    "type": "array",
                    "description": (
                        "Explicit relationships to other objects. "
                        'E.g. [{"target": "CounterTop", "type": "on"}, '
                        '{"target": "Kitchen", "type": "in"}]. '
                        "Common types: on, in, near, contains, is_a."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "target": {
                                "type": "string",
                                "description": "Name of the related object.",
                            },
                            "type": {
                                "type": "string",
                                "description": "Relation type, e.g. 'on', 'in', 'near'.",
                            },
                        },
                        "required": ["target", "type"],
                    },
                },
                "properties": {
                    "type": "object",
                    "description": (
                        "Key-value pairs for observable state, "
                        'e.g. {"is_open": true, "held_by_agent": false}.'
                    ),
                },
            },
            required=["name"],
        )

    def __call__(
        self,
        memory: MemorySystem,
        name: str,
        node_type: str = "object",
        relations: list = None,
        properties: dict = None,
        scope: str = None,
    ) -> str:
        return memory.add_spatial(
            name=name,
            node_type=node_type,
            relations=relations,
            properties=properties,
            scope=scope,
        )


class RememberEventTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="remember_event",
            description=(
                "Record an action the agent just attempted, along with its outcome. "
                "Call this after every significant action to build a history of what has been tried."
            ),
            parameters={
                "action": {
                    "type": "string",
                    "description": "Description of the action taken, e.g. 'Pick up Apple from CounterTop'.",
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the action succeeded.",
                },
                "feedback": {
                    "type": "string",
                    "description": "Environment feedback or error message, e.g. 'Object not reachable'.",
                },
                "note": {
                    "type": "string",
                    "description": "Optional observation or reasoning note about why it succeeded or failed.",
                },
            },
            required=["action", "success"],
        )

    def __call__(
        self,
        memory: MemorySystem,
        action: str,
        success: bool,
        feedback: str = "",
        note: str = "",
        metadata: dict = None,
    ) -> str:
        return memory.add_event(
            action=action,
            success=success,
            feedback=feedback,
            note=note,
            metadata=metadata,
        )


class RememberSceneTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="remember_scene",
            description=(
                "Save a snapshot of the current visual scene. "
                "Call this when entering a new area or when the scene changes significantly."
            ),
            parameters={
                "caption": {
                    "type": "string",
                    "description": "Natural language description of the current scene.",
                },
                "visible_objects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": 'List of object names currently visible, e.g. ["Sink", "Faucet", "Mug"].',
                },
            },
            required=["caption"],
        )

    def __call__(
        self,
        memory: MemorySystem,
        caption: str,
        visible_objects: list = None,
        scope: str = None,
    ) -> str:
        return memory.add_scene(
            caption=caption, visible_objects=visible_objects, scope=scope
        )


class UpdateObjectTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="update_object",
            description=(
                "Update or create a spatial object in memory. "
                "Unlike remember_object, this REPLACES all existing location relations "
                "with the new ones provided — no need to forget first. "
                "Use this after any action that changes an object's location or state "
                "(picked up, placed, moved, opened, closed)."
            ),
            parameters={
                "name": {
                    "type": "string",
                    "description": "Object name, e.g. 'apple', 'fridge'.",
                },
                "relations": {
                    "type": "array",
                    "description": (
                        "New location relationships, REPLACING all existing ones. "
                        'E.g. [{"target": "fridge", "type": "in"}]. '
                        "Common types: on, in, near, contains."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "target": {
                                "type": "string",
                                "description": "Related object name.",
                            },
                            "type": {"type": "string", "description": "Relation type."},
                        },
                        "required": ["target", "type"],
                    },
                },
                "properties": {
                    "type": "object",
                    "description": (
                        "Key-value state properties to merge, "
                        'e.g. {"is_open": true, "delivered": true}.'
                    ),
                },
                "node_type": {
                    "type": "string",
                    "description": "Object category.",
                    "enum": ["object", "receptacle", "room", "landmark"],
                },
            },
            required=["name"],
        )

    def __call__(
        self,
        memory: MemorySystem,
        name: str,
        node_type: str = "object",
        relations: list = None,
        properties: dict = None,
        scope: str = None,
    ) -> str:
        nid = memory.update_spatial(
            name=name,
            node_type=node_type,
            relations=relations,
            properties=properties,
            scope=scope,
        )
        return f"Spatial record for '{name}' updated (id={nid})."


class ForgetObjectTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="forget_object",
            description=(
                "Remove an object from spatial memory. "
                "Use this when an object has moved or its recorded location is no longer valid. "
                "After forgetting, call remember_object to record the updated location."
            ),
            parameters={
                "name": {
                    "type": "string",
                    "description": "Object name to remove, e.g. 'Plate', 'Apple'.",
                },
            },
        )

    def __call__(self, memory: MemorySystem, name: str) -> str:
        ids = memory.spatial.name_index.get(name.lower(), [])
        if not ids:
            return f"'{name}' not found in spatial memory."
        for nid in list(ids):
            memory.spatial.remove_node(nid)
        return f"Removed '{name}' from spatial memory."


class QuerySpatialTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="query_spatial",
            description=(
                "Retrieve objects and locations from spatial memory. "
                "Use this to recall where an object was last seen or what is known about a location."
            ),
            parameters={
                "query": {
                    "type": "string",
                    "description": "Object name or natural language question, e.g. 'Apple' or 'Where is the Knife?'.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 3.",
                },
            },
            required=["query"],
        )

    def __call__(
        self, memory: MemorySystem, query: str, top_k: int = 3, scope: str = None
    ) -> str:
        return _format_query_results(memory.query_spatial(query, top_k, scope=scope))


class QueryEventTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="query_event",
            description=(
                "Retrieve past actions and their outcomes from event memory. "
                "Use this to recall what has been tried, what failed, or what succeeded."
            ),
            parameters={
                "query": {
                    "type": "string",
                    "description": "Action description or keyword, e.g. 'pick up' or 'failed actions'.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 3.",
                },
            },
            required=["query"],
        )

    def __call__(self, memory: MemorySystem, query: str, top_k: int = 3) -> str:
        return _format_query_results(memory.query_event(query, top_k))


class QuerySceneTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="query_scene",
            description=(
                "Retrieve past scene snapshots from scene memory. "
                "Use this to recall what a previous area looked like or what objects were visible."
            ),
            parameters={
                "query": {
                    "type": "string",
                    "description": "Scene description or keyword, e.g. 'kitchen' or 'where I saw the mug'.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default: 3.",
                },
                "scope": {
                    "type": "string",
                    "description": "Optional scene identifier; omit to search all recorded scenes.",
                },
                "max_records": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 12,
                    "description": "Record budget (at most 12); overrides top_k when supplied.",
                },
            },
            required=["query"],
        )

    def __call__(
        self,
        memory: MemorySystem,
        query: str,
        top_k: int = 3,
        scope: str = None,
        max_records: int = None,
    ):
        results = memory.query_scene(
            query, top_k if max_records is None else max_records, scope=scope
        )
        text = _format_query_results(results)
        images = [r["image_path"] for r in results if r.get("image_path")]
        if images:
            return {"text": text, "images": images}
        return text


class QueryExperienceTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="query_experience",
            description=(
                "Retrieve consolidated experience memory inferred from past events. "
                "Use this before decisions that depend on remembered object locations, "
                "household placement habits, failed interactions, locked/blocked targets, "
                "or known successful affordances. This queries the embodied_memorizer typed "
                "experience layer, not raw chat history."
            ),
            parameters={
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language memory query, e.g. 'Where is the cup now?', "
                        "'Where should bottles be stored in this household?', or "
                        "'Which containers are openable or blocked?'."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of consolidated memories to return. Default: 5.",
                },
                "namespace": {
                    "type": "string",
                    "description": (
                        "Optional memory namespace/household id. Leave empty when the "
                        "environment does not provide namespaces."
                    ),
                },
            },
            required=["query"],
        )

    def __call__(
        self,
        memory: MemorySystem,
        query: str,
        top_k: int = 5,
        namespace: str = None,
    ) -> str:
        namespace = namespace or None
        results = memory.query_relevant(
            query,
            namespace=namespace,
            experience_top_k=top_k,
            event_top_k=0,
            spatial_top_k=0,
            scene_top_k=0,
        )
        experience_results = [
            item for item in results if item.get("_layer") == "experience"
        ]
        return _format_query_results(experience_results)


# ── 阶段工具集 ─────────────────────────────────────────────────────────────────

# ── ReAct 两阶段工具集 ────────────────────────────────────────────────────────

# Phase 1（记忆写入/更新）允许的工具
STORE_PHASE_TOOLS: frozenset = frozenset(
    {
        "remember_object",
        "remember_event",
        "remember_scene",
        "forget_object",
        "query_spatial",  # 用于更新前验证已有记录（Rule 3）
        "query_scene",  # 用于判断物体是否为新实例（Rule 2 去重检查）
    }
)

# Phase 2（记忆检索/决策）允许的工具
RETRIEVE_PHASE_TOOLS: frozenset = frozenset(
    {
        "query_spatial",
        "query_event",
        "query_scene",
        "query_experience",
    }
)

# ── Harness 六阶段工具集 ──────────────────────────────────────────────────────

# Step 3（记忆更新）允许的工具：直接 upsert，无需先 forget
HARNESS_UPDATE_TOOLS: frozenset = frozenset(
    {
        "update_object",
        "remember_event",
        "remember_scene",
    }
)

# Step 5（记忆检索）允许的工具
HARNESS_RETRIEVE_TOOLS: frozenset = frozenset(
    {
        "query_spatial",
        "query_event",
        "query_scene",
        "query_experience",
    }
)


# ── 工具注册表 ─────────────────────────────────────────────────────────────────


class MemoryToolRegistry:
    """统一管理所有记忆工具，提供 schema 获取和工具分发。"""

    def __init__(self):
        self.tools: List[BaseTool] = [
            RememberObjectTool(),
            RememberEventTool(),
            RememberSceneTool(),
            ForgetObjectTool(),
            UpdateObjectTool(),
            QuerySpatialTool(),
            QueryEventTool(),
            QuerySceneTool(),
            QueryExperienceTool(),
        ]
        self._tool_map: Dict[str, BaseTool] = {t.name: t for t in self.tools}

    def get_schemas(self) -> List[Dict]:
        return [t.to_schema() for t in self.tools]

    def execute(
        self,
        tool_name: str,
        args: Dict,
        memory: MemorySystem,
        execution_metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if tool_name not in self._tool_map:
            raise ValueError(f"Unknown tool: {tool_name}")
        call_args = deepcopy(args or {})
        trusted_metadata = deepcopy(execution_metadata or {})
        if tool_name in {"remember_object", "update_object", "query_spatial"}:
            call_args.pop("scope", None)
            call_args.pop("scene", None)
            if (
                trusted_metadata.get("spatial_identity_scope_enabled", True)
                and trusted_metadata.get("scene") is not None
            ):
                call_args["scope"] = trusted_metadata["scene"]
        if tool_name == "remember_event":
            call_args.pop("execution_metadata", None)
            call_args.pop("metadata", None)
            trusted_metadata.setdefault("raw_action_text", call_args.get("action"))
            trusted_metadata.setdefault("raw_feedback_text", call_args.get("feedback"))
            call_args["metadata"] = trusted_metadata
        elif tool_name == "remember_scene":
            # Scope, like the RGB path, comes from the harness, not model text.
            call_args["scope"] = trusted_metadata.get("scene")
        elif tool_name == "query_experience":
            namespace = trusted_metadata.get("memory_namespace")
            if namespace:
                call_args["namespace"] = namespace
        return self._tool_map[tool_name](memory, **call_args)

    def get_descriptions(self) -> str:
        """将工具 schema 渲染为注入 prompt 的文本描述。"""
        lines = ["## Available Memory Tools\n"]
        for tool in self.tools:
            lines.append(f"### {tool.name}")
            lines.append(tool.description)
            required = tool.required
            for param, schema in tool.parameters.items():
                req_mark = " (required)" if param in required else ""
                desc = schema.get("description", "")
                lines.append(f"- {param}{req_mark}: {desc}")
            lines.append("")
        return "\n".join(lines)


# ── 内部辅助函数 ───────────────────────────────────────────────────────────────


def _format_query_results(results: List[dict]) -> str:
    """将 memory_query 结果格式化为文本，供注入 prompt 使用。"""
    if not results:
        return "No relevant memories found."
    lines = []
    for r in results:
        layer = r.get("_layer", "unknown")
        if layer == "spatial":
            props = r.get("properties", {})
            props_str = ""
            if props:
                props_str = " [" + ", ".join(f"{k}={v}" for k, v in props.items()) + "]"
            lines.append(f"[Object] {r.get('name', '?')}{props_str}")
            for e in r.get("_edges", [])[:3]:
                lines.append(f"  -> {e['relation']} {e['target_name']}")
        elif layer == "event":
            status = "SUCCESS" if r.get("success") else "FAILED"
            lines.append(
                f"[Event] Step {r.get('step', '?')}: {r.get('action', '?')} [{status}]"
            )
            if r.get("feedback"):
                lines.append(f"  Feedback: {r['feedback']}")
        elif layer == "scene":
            visible = ", ".join(r.get("visible_objects", []))
            line = f"[Scene] Step {r.get('step', '?')}: {r.get('caption', '?')}"
            line += (
                f" | id={r.get('id', '?')} | scope={r.get('scope') or 'unspecified'}"
            )
            if visible:
                line += f" | Visible: {visible}"
            lines.append(line)
        elif layer in {"experience", "habit"}:
            if r.get("experience_type") == "observed_object_location":
                lines.append(
                    f"[Location] {r.get('object', '?')} latest known location: "
                    f"{r.get('relation', 'on_or_in')} {r.get('target', '?')} "
                    f"(score={r.get('score', '?')}; observations={r.get('observation_count', '?')}; "
                    f"last_step={r.get('last_step', '?')})"
                )
                history = r.get("location_history") or []
                if len(history) > 1:
                    compact = []
                    for item in history[:3]:
                        compact.append(
                            f"step {item.get('step', '?')}: "
                            f"{item.get('relation', 'on_or_in')} {item.get('target', '?')}"
                        )
                    lines.append(f"  Recent location history: {'; '.join(compact)}")
                for evidence in r.get("evidence", [])[:2]:
                    lines.append(f"  Evidence: {evidence}")
                continue
            if r.get("experience_type") == "interaction_failure_constraint":
                modes = ", ".join(
                    f"{mode}:{count}"
                    for mode, count in (r.get("failure_modes") or {}).items()
                )
                superseded = (
                    "; superseded_by_success" if r.get("superseded_by_success") else ""
                )
                status = (
                    f"; status={r.get('current_status')}"
                    if r.get("current_status")
                    else ""
                )
                lines.append(
                    f"[Constraint] Avoid {r.get('action_type', 'Interact')} on "
                    f"{r.get('target', '?')} "
                    f"(score={r.get('score', '?')}; support={r.get('support_count', '?')}; "
                    f"failure_modes={modes or 'failed'}{status}{superseded})"
                )
                for evidence in r.get("evidence", [])[:3]:
                    lines.append(f"  Evidence: {evidence}")
                continue
            if r.get("experience_type") == "interaction_affordance":
                conflicted = (
                    "; conflicted_by_failure" if r.get("conflicted_by_failure") else ""
                )
                status = (
                    f"; status={r.get('current_status')}"
                    if r.get("current_status")
                    else ""
                )
                lines.append(
                    f"[Affordance] {r.get('action_type', 'Interact')} worked on "
                    f"{r.get('target', '?')} "
                    f"(score={r.get('score', '?')}; support={r.get('support_count', '?')}"
                    f"{status}{conflicted})"
                )
                for evidence in r.get("evidence", [])[:3]:
                    lines.append(f"  Evidence: {evidence}")
                continue
            if r.get("experience_type") == "interaction_state":
                blocked = ", ".join(r.get("blocked_action_types") or [])
                available = ", ".join(r.get("available_action_types") or [])
                policy = (
                    f"blocked_actions={blocked}"
                    if blocked
                    else f"available_actions={available or r.get('action_type', 'Interact')}"
                )
                lines.append(
                    f"[InteractionState] {r.get('action_type', 'Interact')} on "
                    f"{r.get('target', '?')} is {r.get('current_status', 'unknown')} "
                    f"({policy}; success={r.get('success_count', 0)}; "
                    f"failure={r.get('failure_count', 0)}; conflicts={r.get('conflict_count', 0)})"
                )
                for evidence in r.get("evidence", [])[:3]:
                    lines.append(f"  Evidence: {evidence}")
                continue
            objects = ", ".join(r.get("support_objects", []))
            lines.append(
                f"[Experience] Corrected placement target: {r.get('target', '?')} "
                f"(score={r.get('score', '?')}; support={r.get('support_count', '?')}; "
                f"support_objects={objects})"
            )
            negatives = ", ".join(r.get("negative_targets", []))
            if negatives:
                lines.append(f"  Failed targets to avoid: {negatives}")
            for trace in r.get("support_traces", [])[:3]:
                lines.append(f"  Semantic support trace: {trace}")
            for evidence in r.get("evidence", [])[:3]:
                lines.append(f"  Evidence: {evidence}")
    return "\n".join(lines)
