"""Memory context building and spatial update helpers.

Provides functions for injecting memory into prompts and automatically
updating spatial memory from environment feedback.
"""

import re
import json
import logging

logger = logging.getLogger("embodied_memorizer")


def build_memory_context(memory, config, step, instruction):
    """Build memory context text for prompt injection.

    Always produces output if any memory exists, not just on failures.
    Uses to_text() methods to include ALL known objects/events/scenes,
    with a token budget to prevent prompt bloat.

    Args:
        memory: MemorySystem instance
        config: Full merged config dict
        step: Current step number
        instruction: Current user instruction (unused but kept for future query-based retrieval)

    Returns:
        Formatted memory text string, or empty string if no memory.
    """
    sections = []
    token_budget = config["prompt"].get("max_memory_tokens", 800)
    current_tokens = 0  # approximate: 1 token ≈ 4 chars

    # 1. Warnings — highest priority, prepend later
    warning_text = ""
    pattern = memory.event.get_action_pattern()
    if pattern["repeated_failures"] or pattern["loop_detected"]:
        warning_lines = ["## Memory: !! Warnings !!"]
        if pattern["loop_detected"]:
            warning_lines.append(
                "- ACTION LOOP DETECTED: You are repeating the same actions. "
                "Change your strategy!"
            )
        if pattern["repeated_failures"]:
            warning_lines.append(
                f"- Repeated failed actions: {', '.join(pattern['repeated_failures'])}. "
                "Try a different approach!"
            )
        warning_text = "\n".join(warning_lines)

    # 2. Agent state — what the agent is currently holding
    agent_state = build_agent_state(memory, step)
    if agent_state:
        est = len(agent_state) // 4
        if current_tokens + est <= token_budget:
            sections.append(agent_state)
            current_tokens += est

    # 3. Consolidated experience memory — lessons inferred from raw events.
    experience_text = build_experience_memory(memory, instruction, config)
    if experience_text:
        est = len(experience_text) // 4
        if current_tokens + est <= token_budget:
            sections.append(experience_text)
            current_tokens += est

    # 4. All known objects (最近插入优先)
    spatial_text = memory.spatial.to_text(
        top_k=config["prompt"].get("spatial_top_k", 10),
    )
    if "No spatial memory" not in spatial_text:
        spatial_text = spatial_text.replace(
            "[Spatial Memory]", "## Memory: Known Objects & Locations"
        )
        est = len(spatial_text) // 4
        if current_tokens + est <= token_budget:
            sections.append(spatial_text)
            current_tokens += est

    # 5. 与当前指令相关的历史动作（Level 1 关键词检索，embedding 兜底在 RetrievalEngine）
    event_text = memory.event.to_text(
        query=instruction,
        top_k=config["prompt"].get("event_top_k", 5),
    )
    if "No event memory" not in event_text:
        event_text = event_text.replace("[Event Memory]", "## Memory: Recent Actions")
        est = len(event_text) // 4
        if current_tokens + est <= token_budget:
            sections.append(event_text)
            current_tokens += est

    # 6. 与当前指令相关的场景快照（两级检索，低优先级）
    scene_text = memory.scene.to_text(
        query=instruction,
        top_k=config["prompt"].get("scene_top_k", 2),
    )
    if "No scene memory" not in scene_text:
        scene_text = scene_text.replace("[Scene Memory]", "## Memory: Scene Context")
        est = len(scene_text) // 4
        if current_tokens + est <= token_budget:
            sections.append(scene_text)
            current_tokens += est

    # Prepend warnings at top (highest priority)
    if warning_text:
        sections.insert(0, warning_text)

    if not sections:
        return ""

    return "\n\n".join(sections)


def build_agent_state(memory, step):
    """Build a one-line agent state summary.

    Args:
        memory: MemorySystem instance
        step: Current step number

    Returns:
        Formatted agent state string, or empty string.
    """
    state = memory.get_agent_state()
    parts = []
    if state["held_objects"]:
        parts.append(f"Currently holding: {', '.join(state['held_objects'])}")
    else:
        if step > 0:
            parts.append("Currently holding: nothing")
    if not parts:
        return ""
    return "## Agent State\n" + "\n".join(f"- {p}" for p in parts)


def build_experience_memory(memory, instruction, config):
    query_relevant = getattr(memory, "query_relevant", None)
    if not callable(query_relevant):
        return ""
    results = query_relevant(
        instruction,
        experience_top_k=config["prompt"].get("experience_top_k", 5),
        event_top_k=3,
        spatial_top_k=0,
        scene_top_k=0,
    )
    experiences = [item for item in results if item.get("_layer") == "experience"]
    if not experiences:
        return ""
    lines = ["## Memory: Consolidated Experience"]
    for item in experiences:
        if item.get("experience_type") == "observed_object_location":
            lines.append(
                f"- Latest known location: {item.get('object', '?')} "
                f"{item.get('relation', 'on_or_in')} {item.get('target', '?')} "
                f"(observations={item.get('observation_count', '?')}; "
                f"last_step={item.get('last_step', '?')})"
            )
            history = item.get("location_history") or []
            if len(history) > 1:
                compact = [
                    f"step {h.get('step', '?')}: {h.get('relation', 'on_or_in')} {h.get('target', '?')}"
                    for h in history[:3]
                ]
                lines.append(f"  Recent location history: {'; '.join(compact)}")
            continue
        if item.get("experience_type") == "interaction_failure_constraint":
            modes = ", ".join(
                f"{mode}:{count}"
                for mode, count in (item.get("failure_modes") or {}).items()
            )
            superseded = (
                "; superseded_by_success" if item.get("superseded_by_success") else ""
            )
            status = (
                f"; status={item.get('current_status')}"
                if item.get("current_status")
                else ""
            )
            lines.append(
                f"- Interaction constraint: avoid {item.get('action_type', 'Interact')} "
                f"on {item.get('target', '?')} "
                f"(support={item.get('support_count', '?')}; "
                f"failure_modes={modes or 'failed'}{status}{superseded})"
            )
            for evidence in item.get("evidence", [])[:2]:
                lines.append(f"  Evidence: {evidence}")
            continue
        if item.get("experience_type") == "interaction_affordance":
            conflicted = (
                "; conflicted_by_failure" if item.get("conflicted_by_failure") else ""
            )
            status = (
                f"; status={item.get('current_status')}"
                if item.get("current_status")
                else ""
            )
            lines.append(
                f"- Interaction affordance: {item.get('action_type', 'Interact')} "
                f"worked on {item.get('target', '?')} "
                f"(support={item.get('support_count', '?')}{status}{conflicted})"
            )
            for evidence in item.get("evidence", [])[:2]:
                lines.append(f"  Evidence: {evidence}")
            continue
        if item.get("experience_type") == "interaction_state":
            blocked = ", ".join(item.get("blocked_action_types") or [])
            available = ", ".join(item.get("available_action_types") or [])
            policy = (
                f"blocked_actions={blocked}"
                if blocked
                else f"available_actions={available or item.get('action_type', 'Interact')}"
            )
            lines.append(
                f"- Interaction state: {item.get('action_type', 'Interact')} "
                f"on {item.get('target', '?')} is {item.get('current_status', 'unknown')} "
                f"({policy}; success={item.get('success_count', 0)}; "
                f"failure={item.get('failure_count', 0)}; conflicts={item.get('conflict_count', 0)})"
            )
            for evidence in item.get("evidence", [])[:2]:
                lines.append(f"  Evidence: {evidence}")
            continue
        objects = ", ".join(item.get("support_objects", []))
        negatives = ", ".join(item.get("negative_targets", []))
        lines.append(
            f"- Corrected placement target: {item.get('target', '?')} "
            f"(support={item.get('support_count', '?')}; objects={objects})"
        )
        if negatives:
            lines.append(f"  Failed targets to avoid: {negatives}")
        for trace in item.get("support_traces", [])[:3]:
            lines.append(f"  Semantic support trace: {trace}")
        for evidence in item.get("evidence", [])[:2]:
            lines.append(f"  Evidence: {evidence}")
    return "\n".join(lines)


# ── Spatial update patterns ──

_OBJECT_PATTERNS = [
    r"find (?:a |the )?(.+?)(?:\s*$)",
    r"pick up (?:a |the )?(.+?)(?:\s*$)",
    r"put down (?:a |the )?(.+?)(?:\s+(?:on|in|to)\s+(.+))?(?:\s*$)",
    r"open (?:a |the )?(.+?)(?:\s*$)",
    r"close (?:a |the )?(.+?)(?:\s*$)",
    r"turn on (?:a |the )?(.+?)(?:\s*$)",
    r"turn off (?:a |the )?(.+?)(?:\s*$)",
    r"slice (?:a |the )?(.+?)(?:\s*$)",
    r"navigate to (?:a |the )?(.+?)(?:\s*$)",
    r"Navigation (?:a |the )?(.+?)(?:\s*$)",
]


def auto_update_spatial(memory, action_desc, feedback, success, step):
    """Extract object information from action and feedback, update spatial memory.

    Args:
        memory: MemorySystem instance
        action_desc: Action description string
        feedback: Environment feedback string
        success: Whether the action succeeded
        step: Current step number
    """
    action_lower = action_desc.lower()

    for pattern in _OBJECT_PATTERNS:
        match = re.search(pattern, action_desc, re.IGNORECASE)
        if match:
            obj_name = match.group(1).strip()
            if obj_name:
                props = {}
                if "pick up" in action_lower and success:
                    props["held_by_agent"] = True
                elif "put down" in action_lower and success:
                    props["held_by_agent"] = False
                    if match.lastindex and match.lastindex >= 2 and match.group(2):
                        location = match.group(2).strip()
                        memory.add_spatial(
                            name=obj_name,
                            properties=props,
                            relations=[{"target": location, "type": "on"}],
                        )
                        return
                elif "open" in action_lower and success:
                    props["is_open"] = True
                elif "close" in action_lower and success:
                    props["is_open"] = False
                elif "turn on" in action_lower and success:
                    props["is_on"] = True
                elif "turn off" in action_lower and success:
                    props["is_on"] = False
                elif "slice" in action_lower and success:
                    props["is_sliced"] = True

                memory.add_spatial(name=obj_name, properties=props)
            break


def should_save_scene(step):
    """Determine if current scene should be saved."""
    # return step % 5 == 0 or step == 0
    return True  # For testing, save every step


def auto_save_scene(memory, observation, reasoning, step):
    """Auto-save scene snapshot.

    Args:
        memory: MemorySystem instance
        observation: Image path or observation dict
        reasoning: Model reasoning output (JSON string)
        step: Current step number
    """
    image_path = None
    if isinstance(observation, str):
        image_path = observation

    caption = ""
    try:
        parsed = json.loads(reasoning)
        caption = parsed.get("visual_state_description", "")
    except (json.JSONDecodeError, AttributeError):
        pass

    if caption or image_path:
        memory.add_scene(
            caption=caption or f"Scene at step {step}", image_path=image_path
        )
