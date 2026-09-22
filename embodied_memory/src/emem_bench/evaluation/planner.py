"""OpenAI-compatible planner used by the benchmark API runner."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .matchers import object_type


API_BALANCE_MARKERS = (
    "insufficient balance",
    "insufficient quota",
    "insufficient funds",
    "billing",
    "payment required",
    "insufficient credit",
    "exceeded your current quota",
    "insufficient_quota",
    "http 402",
)


class APIBalanceError(RuntimeError):
    """Raised immediately for explicit API billing or quota-exhaustion failures."""


class APITransportError(RuntimeError):
    """No model response was received after the configured API retry budget."""


class LocalAPIServiceUnavailableError(APITransportError):
    """A local loopback model service disappeared; do not score the episode."""


def is_api_balance_error(error: BaseException | str) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in API_BALANCE_MARKERS)


def is_loopback_service_unavailable(
    *,
    base_url: str | None,
    error: BaseException | str,
) -> bool:
    """Recognize an unavailable local vLLM endpoint without affecting remote APIs."""
    return (
        should_bypass_proxy(chat_url(base_url))
        and "[errno 111] connection refused" in str(error).lower()
    )


def is_retryable_proxy_invalid_argument(error: BaseException | str) -> bool:
    """Recognize the proxy's transient outer-429/inner-400 transport failure."""
    text = str(error).lower()
    return (
        "http 429" in text
        and "request contains an invalid argument" in text
        and "code" in text
        and "400" in text
    )


SYSTEM_PROMPT = """You are controlling an embodied household robot in AI2-THOR.
Choose exactly one next action from the provided available_actions.
The available_actions list is the complete action space for the current step.
It contains currently executable navigation and interaction targets.
Pickupable objects may expose Navigate actions, and PickUp is exposed only when the object
is expected to be visible and reachable in the current state.
Before picking up an object, navigate to it or open the relevant container when needed.
PutObject only works while the agent is holding an object; if holding is null/nothing,
pick up the task object before choosing any PutObject action.
If a PickUp attempt fails, do not repeat the same PickUp unless the observation has changed
because you navigated, moved, rotated, or opened the relevant container.
If a Navigate, Open, Close, or TransferContents action just succeeded, do not repeat
the same action on the same target unless the current observation clearly changed the task state.
Only choose Done after the task objective is satisfied. For placement or storage
instructions, a successful PickUp is only an intermediate step; continue until the object
is placed in the inferred target location. For retrieval-only instructions, Done after
pickup can be appropriate.
For hide/store instructions involving an openable container, after a successful PutObject
prefer Close on that same container before Done when Close is available.
Do not invent object names, object ids, or action ids outside available_actions.
Other actions do not move the robot unless their text explicitly says Navigate.
Use the context trajectory to avoid repeated failed behaviors.
When context feedback mentions Preference Violation, HumanIntervention, corrected behavior,
or a corrected placement, infer the household rule from that evidence and follow it.
If the memory context contains a failed placement followed by human intervention or a
corrected placement, treat the corrected placement as the primary evidence and ignore
unrelated successful placements of other objects as weak priors.
When memory_context contains embodied_memorizer Retrieved Events, prioritize those event
records over commonsense guesses or unrelated spatial-memory entries.
When memory_context contains embodied_memorizer Latest Placement Facts, treat those as
the highest-priority current object-location evidence.
The context trajectory is past memory only; actions in it do not complete the current probe.
Use the Done action only when the current probe is complete in the current state or no useful action remains.
Return ONLY a JSON object in this format:
{"action_id": 0, "reasoning": "brief reason"}
Do not return markdown, prose, or multiple actions."""


SEQUENCE_SYSTEM_PROMPT = """You are controlling an embodied household robot in AI2-THOR.
Plan a short executable action sequence for the current probe.
The initial available_actions list is the action space for the current state.
Future action ids may change after the environment executes earlier actions, so prefer
semantic action objects with action_type, target_label, and target_type. You may include
action_id for the first step when it is visible in available_actions.
Use Navigate before PickUp when needed. Do not choose PutObject before the robot is holding
the task object. For placement or storage instructions, continue until the object is placed
in the inferred target location. For retrieval-only instructions, picking up the target object
can complete the task. Include Done as the final action only when the objective should be
satisfied after the preceding actions.
Use the memory_context as past evidence. If it contains failed placements followed by
human intervention or corrected placements, infer the household rule from that evidence.
Return ONLY a JSON object in this format:
{"actions": [{"action_type": "Navigate", "target_label": "Fridge", "target_type": "Fridge"}], "reasoning": "brief reason"}
Do not return markdown or prose."""


OBJECT_ID_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_]*)"
    r"(?:\|[+-]?\d+(?:\.\d+)?){1,3}"
    r"(?:___[A-Za-z0-9_]+)?"
    r"(?:\|[A-Za-z0-9_]+)?"
)
HAN_RE = re.compile(r"[\u4e00-\u9fff]")

SEQUENCE_ACTION_TYPE_ALIASES = {
    "navigate": "Navigate",
    "pickup": "PickUp",
    "pickupobject": "PickUp",
    "put": "PutObject",
    "putobject": "PutObject",
    "place": "PutObject",
    "open": "Open",
    "openobject": "Open",
    "close": "Close",
    "closeobject": "Close",
    "toggleon": "ToggleOn",
    "toggleobjecton": "ToggleOn",
    "toggleoff": "ToggleOff",
    "toggleobjectoff": "ToggleOff",
    "transfercontents": "TransferContents",
    "rotateleft": "RotateLeft",
    "rotateright": "RotateRight",
    "moveforward": "MoveForward",
    "memoryinsufficient": "MemoryInsufficient",
    "abstain": "MemoryInsufficient",
    "done": "Done",
    "finish": "Done",
    "stop": "Done",
}


def canonicalize_sequence_action(action: Any) -> Any:
    """Normalize common simulator action aliases before execution or SFT export."""
    if not isinstance(action, dict):
        return action
    raw_type = action.get("action_type") or action.get("action")
    if not raw_type:
        return action
    alias_key = re.sub(r"[^A-Za-z0-9]+", "", str(raw_type)).lower()
    canonical_type = SEQUENCE_ACTION_TYPE_ALIASES.get(alias_key)
    if canonical_type is None:
        return action
    normalized = dict(action)
    normalized["action_type"] = canonical_type
    normalized.pop("action", None)
    return normalized


def strip_object_ids(value: Any) -> Any:
    if isinstance(value, str):
        return OBJECT_ID_RE.sub(r"\1", value)
    return value


def sanitize_context_description(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = strip_object_ids(value)
    if text.startswith("## embodied_memorizer Retrieved"):
        return text
    if "corrected by human intervention" in text or "homeowner-preferred" in text:
        return "Executed household tasks; one placement failed and was corrected by human intervention."
    return text


def sanitize_context_action_text(value: Any, action_type: str | None = None) -> Any:
    if not isinstance(value, str):
        return value
    text = strip_object_ids(value)
    if action_type == "HumanIntervention" or text.startswith("Human intervention:"):
        return "Human intervention: the failed placement was corrected."
    if HAN_RE.search(text):
        action_templates = {
            "Navigate": "Navigate to target.",
            "Open": "Open target.",
            "Close": "Close target.",
            "PickUp": "Pick up target object.",
            "PickupObject": "Pick up target object.",
            "PutObject": "Place the held object at target.",
            "ToggleOn": "Toggle target on.",
            "ToggleOff": "Toggle target off.",
            "Slice": "Slice target.",
            "Clean": "Clean target.",
        }
        return action_templates.get(
            str(action_type or ""), "Execute the context action."
        )
    text = text.replace(
        "the homeowner-preferred storage location", "the corrected storage location"
    )
    text = text.replace("homeowner-preferred", "corrected")
    text = text.replace(" (correct behavior)", "")
    return text


def sanitize_context_feedback(
    value: Any,
    success: Any = None,
    *,
    preserve_durable_failure: bool = False,
) -> Any:
    if not isinstance(value, str):
        return value
    text = strip_object_ids(value)
    if success is False and text.startswith("Preference Violation:"):
        return "Preference violation: this placement did not match the household habit."
    if success is False and text.startswith("Action Failed:"):
        if preserve_durable_failure:
            lowered = text.lower()
            if "locked" in lowered or "cannot be opened" in lowered:
                return "Action failed: locked."
            if (
                "invalid placement" in lowered
                or "not a valid receptacle" in lowered
                or "cannot be placed" in lowered
            ):
                return "Action failed: invalid placement."
        return "Action failed: this interaction was not accepted."
    if text.startswith("Human intervention:"):
        return "Human intervention: the failed placement was corrected."
    if HAN_RE.search(text):
        if success is True:
            return "Action succeeded."
        if success is False:
            return "Action failed."
        return "Context feedback was observed."
    return text


def image_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def chat_url(base_url: str | None) -> str:
    root = (
        base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    ).rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    return f"{root}/chat/completions"


def should_bypass_proxy(url: str) -> bool:
    host = urllib.parse.urlparse(url).hostname
    if host is None:
        return False
    return host == "localhost" or host == "::1" or host.startswith("127.")


def read_http_response_with_deadline(response: Any, timeout: int) -> bytes:
    """Read a non-streaming response without allowing trickle bytes to extend its deadline."""
    deadline = time.monotonic() + timeout
    chunks: list[bytes] = []
    read1 = getattr(response, "read1", None)
    socket = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"HTTP response read exceeded {timeout}s deadline")
        if socket is not None:
            current_timeout = socket.gettimeout()
            if current_timeout is None or current_timeout > remaining:
                socket.settimeout(remaining)
        try:
            chunk = read1(65536) if callable(read1) else response.read(1)
        except TimeoutError as exc:
            raise TimeoutError(
                f"HTTP response read exceeded {timeout}s deadline"
            ) from exc
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def post_chat_completion(
    *,
    model_name: str,
    base_url: str | None,
    api_key: str | None,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    timeout: int,
    retries: int,
    retry_delay: float,
    response_format: dict[str, Any] | None = None,
    metadata_sink: dict[str, Any] | None = None,
) -> str:
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = chat_url(base_url)
    opener = None
    if should_bypass_proxy(url):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_error: Exception | None = None
    started = time.perf_counter()

    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            if opener is None:
                response_context = urllib.request.urlopen(req, timeout=timeout)
            else:
                response_context = opener.open(req, timeout=timeout)
            with response_context as resp:
                data = json.loads(
                    read_http_response_with_deadline(resp, timeout).decode("utf-8")
                )
            content = data["choices"][0]["message"].get("content") or ""
            if not content.strip():
                raise RuntimeError("API returned empty message content")
            if metadata_sink is not None:
                usage = data.get("usage") or {}
                metadata_sink.clear()
                metadata_sink.update(
                    {
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                        "latency_seconds": round(time.perf_counter() - started, 6),
                        "attempts": attempt + 1,
                        "response_id": data.get("id"),
                        "served_model": data.get("model"),
                    }
                )
            return content
        except urllib.error.HTTPError as exc:
            try:
                body_text = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body_text = ""
            last_error = (
                RuntimeError(f"HTTP {exc.code}: {body_text[:500]}")
                if body_text
                else exc
            )
            if is_api_balance_error(
                last_error
            ) and not is_retryable_proxy_invalid_argument(last_error):
                raise APIBalanceError(str(last_error)) from exc
            if attempt >= retries:
                break
            time.sleep(retry_delay)
        except Exception as exc:
            if isinstance(exc, APIBalanceError):
                raise
            if is_api_balance_error(exc):
                raise APIBalanceError(str(exc)) from exc
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_delay)

    if last_error is not None and is_api_balance_error(last_error):
        raise APIBalanceError(str(last_error)) from last_error
    failure = APITransportError(
        f"API call failed after {retries + 1} attempt(s): {last_error}"
    )
    if is_loopback_service_unavailable(base_url=base_url, error=failure):
        raise LocalAPIServiceUnavailableError(str(failure)) from last_error
    raise failure


def compact_context(
    contexts: list[dict],
    max_steps_per_session: int,
    object_labels: dict[str, str] | None = None,
) -> list[dict]:
    """Keep only model-visible history fields.

    This serializes the model-visible observation fields. The model should
    infer habits from actions and feedback rather than answer labels.
    """
    compact = []
    for ctx in contexts:
        steps = []
        for step in ctx.get("steps", [])[:max_steps_per_session]:
            action = step.get("action") or {}
            feedback = step.get("feedback") or {}
            target = action.get("target")
            step_record = {
                "step_id": step.get("step_id"),
                "action": {
                    "action_type": action.get("action_type"),
                    "target_label": (object_labels or {}).get(target)
                    or (object_type(target) if isinstance(target, str) else None),
                    "target_type": object_type(target)
                    if isinstance(target, str)
                    else None,
                    "natural_language": sanitize_context_action_text(
                        action.get("natural_language"),
                        action.get("action_type"),
                    ),
                },
                "success": feedback.get("success"),
                "feedback": sanitize_context_feedback(
                    feedback.get("message"),
                    feedback.get("success"),
                ),
            }
            memory_cue = step.get("memory_cue_exposed")
            if memory_cue:
                step_record["observation_note"] = sanitize_context_feedback(
                    memory_cue.get("description"),
                    True,
                )
            steps.append(step_record)
        compact.append(
            {
                "session_name": ctx.get("session_name"),
                "description": sanitize_context_description(ctx.get("description")),
                "total_steps": ctx.get("total_steps"),
                "steps": steps,
            }
        )
    return compact


def compact_held_object(obs: dict) -> str | None:
    state = dict(obs.get("agent_state") or {})
    holding = state.get("holding")
    if not holding:
        return None
    return obs.get("object_labels", {}).get(holding) or holding.split("|")[0]


def compact_agent_state(obs: dict) -> dict:
    return {"holding": compact_held_object(obs)}


def compact_visible_objects(obs: dict, max_objects: int) -> list[dict]:
    objects = []
    for obj in obs.get("visible_objects", [])[:max_objects]:
        objects.append(
            {
                "object_label": obj.get("object_label"),
                "object_type": obj.get("object_type") or obj.get("objectType"),
                "distance": obj.get("distance"),
            }
        )
    return objects


def compact_available_actions(actions: list[dict]) -> list[dict]:
    """Hide raw AI2-THOR objectIds from the model-facing action list.

    The runner asks for action_id, so the model does not need coordinate-bearing
    objectIds. Keeping only labels makes the API prompt closer to EmbodiedBench's
    language-skill action space while env.step still has the private objectId.
    """
    compact = []
    for action in actions:
        compact.append(
            {
                "action_id": action.get("action_id"),
                "action_type": action.get("action_type"),
                "target_label": action.get("target_label"),
                "target_type": action.get("target_type"),
            }
        )
    return compact


_COMPACT_PROMPT_KEYS = {
    "session_name": "n",
    "description": "d",
    "total_steps": "t",
    "steps": "s",
    "step_id": "i",
    "action": "a",
    "action_type": "y",
    "target_label": "l",
    "target_type": "k",
    "natural_language": "x",
    "success": "q",
    "feedback": "f",
    "observation_note": "o",
    "object_label": "l",
    "object_type": "k",
    "distance": "d",
    "step": "s",
    "model_response": "r",
    "reasoning": "z",
    "thought": "z",
    "planner_output_mode": "u",
    "sequence_plan_id": "g",
    "sequence_index": "j",
    "actions": "b",
    "action_id": "i",
}


def _compact_prompt_payload(value: Any, *, top_level: bool = True) -> Any:
    """Shorten repeated nested JSON keys without removing any values."""
    if isinstance(value, list):
        return [_compact_prompt_payload(item, top_level=False) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        (
            key if top_level else _COMPACT_PROMPT_KEYS.get(key, key)
        ): _compact_prompt_payload(item, top_level=False)
        for key, item in value.items()
    }


def compact_semantic_available_actions(
    actions: list[dict],
    *,
    visible_objects: list[dict] | None = None,
    max_navigate_actions: int = 24,
) -> tuple[list[dict], dict]:
    """Bound redundant Navigate instances while preserving executable semantics.

    The model emits stable action_type/target fields and the private executor
    resolves them against the full dynamic action space.  Routed tool-call
    prompts therefore do not need hundreds of instance-level Navigate rows.
    """
    if max_navigate_actions < 1:
        raise ValueError("max_navigate_actions must be >= 1")

    compact = compact_available_actions(actions)

    def semantic_keys(item: dict) -> set[str]:
        return {
            str(value).strip().casefold()
            for value in (item.get("target_label"), item.get("target_type"))
            if value is not None and str(value).strip()
        }

    visible_keys: set[str] = set()
    for item in visible_objects or []:
        if isinstance(item, dict):
            visible_keys.update(
                str(value).strip().casefold()
                for value in (item.get("object_label"), item.get("object_type"))
                if value is not None and str(value).strip()
            )

    non_navigate: list[dict] = []
    navigate: list[dict] = []
    anchor_keys = set(visible_keys)
    terminal_count = 0
    for action in compact:
        action_type = str(action.get("action_type") or "")
        semantic = {
            key: action[key]
            for key in ("action_type", "target_label", "target_type")
            if action.get(key) is not None
        }
        if action_type in {"Done", "Stop", "Finish"}:
            terminal_count += 1
            continue
        if action_type == "Navigate":
            navigate.append(semantic)
        else:
            non_navigate.append(semantic)
            anchor_keys.update(semantic_keys(semantic))

    def deduplicate(items: list[dict]) -> list[dict]:
        output: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for item in items:
            signature = (
                str(item.get("action_type") or ""),
                str(item.get("target_label") or ""),
                str(item.get("target_type") or ""),
            )
            if signature in seen:
                continue
            seen.add(signature)
            output.append(item)
        return output

    non_navigate = deduplicate(non_navigate)
    navigate = deduplicate(navigate)
    anchored = [item for item in navigate if semantic_keys(item) & anchor_keys]
    remaining = [item for item in navigate if not semantic_keys(item) & anchor_keys]

    selected = list(anchored)
    remaining_slots = max(max_navigate_actions - len(selected), 0)
    diverse: list[dict] = []
    seen_types = {
        str(item.get("target_type") or "").strip().casefold()
        for item in selected
        if item.get("target_type")
    }
    for item in remaining:
        target_type = str(item.get("target_type") or "").strip().casefold()
        if target_type and target_type not in seen_types:
            diverse.append(item)
            seen_types.add(target_type)
            if len(diverse) >= remaining_slots:
                break
    selected.extend(diverse)
    selected_signatures = {
        (
            str(item.get("action_type") or ""),
            str(item.get("target_label") or ""),
            str(item.get("target_type") or ""),
        )
        for item in selected
    }
    if len(selected) < max_navigate_actions:
        for item in remaining:
            signature = (
                str(item.get("action_type") or ""),
                str(item.get("target_label") or ""),
                str(item.get("target_type") or ""),
            )
            if signature in selected_signatures:
                continue
            selected.append(item)
            selected_signatures.add(signature)
            if len(selected) >= max_navigate_actions:
                break

    shown = non_navigate + selected
    summary = {
        "total_actions": len(compact),
        "shown_actions": len(shown),
        "total_navigate_targets": len(navigate),
        "shown_navigate_targets": len(selected),
        "omitted_navigate_targets": max(len(navigate) - len(selected), 0),
        "omitted_terminal_pseudo_actions": terminal_count,
        "note": (
            "The private executor retains the full dynamic action space and "
            "binds semantic action_type/target fields after every state change."
        ),
    }
    return shown, summary


def compact_history_action(action: object) -> object:
    if isinstance(action, int):
        return {"action_id": action}
    if not isinstance(action, dict):
        return strip_object_ids(action)
    compact: dict[str, object] = {}
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
            compact[key] = strip_object_ids(value)
    return compact or str(action.get("action") or action)


def compact_history_response(payload: object) -> dict:
    if not isinstance(payload, dict):
        return {}
    compact: dict[str, object] = {}
    for key in (
        "reasoning",
        "thought",
        "planner_output_mode",
        "sequence_plan_id",
        "sequence_index",
    ):
        value = payload.get(key)
        if value is not None:
            compact[key] = strip_object_ids(value)
    action = payload.get("action")
    if action is not None:
        compact["action"] = compact_history_action(action)
    actions = payload.get("actions") or payload.get("action_sequence")
    if isinstance(actions, list):
        compact["actions"] = [compact_history_action(item) for item in actions]
    return compact


def compact_history(history: list[dict], limit: int = 8) -> list[dict]:
    compact_steps = []
    for step in history[-limit:]:
        if not isinstance(step, dict):
            continue
        compact_steps.append(
            {
                "step": step.get("step"),
                "action": compact_history_action(step.get("action")),
                "success": bool(step.get("success")),
                "feedback": strip_object_ids(step.get("feedback")),
                "model_response": compact_history_response(step.get("model_response")),
            }
        )
    return compact_steps


def build_user_prompt(
    *,
    obs: dict,
    history: list[dict],
    max_context_steps: int,
    max_visible_objects: int,
    compact_json: bool = False,
) -> str:
    payload = {
        "instruction": obs.get("instruction"),
        "held_object": compact_held_object(obs),
        "memory_context": compact_context(
            obs.get("context", []),
            max_context_steps,
            obs.get("object_labels", {}),
        ),
        "current_visible_objects": compact_visible_objects(obs, max_visible_objects),
        "previous_model_steps": compact_history(history),
        "available_actions": compact_available_actions(
            obs.get("available_actions", [])
        ),
    }
    serialized_payload = _compact_prompt_payload(payload) if compact_json else payload
    compact_legend = (
        " Compact nested JSON keys: n=session, d=description/distance, t=total steps, "
        "s=step, i=id, a=action, y=action type, l=label, k=object type, x=text, "
        "q=success, f=feedback, o=observation note, r=response, z=reasoning, b=actions."
        if compact_json
        else ""
    )
    return (
        "Select the next action. The action_id must be one of available_actions; "
        "this list is the full selectable action space for this step. "
        "The memory_context describes past events only; it is evidence for where objects are "
        "or which interactions failed, but it does not mean the current instruction is already complete. "
        "If memory_context contains preference violations, human interventions, or corrected placements, "
        "use them to infer the current household rule and avoid the same mistake. "
        "If you see a failed placement followed by human intervention or a corrected placement, "
        "treat the corrected placement as the key evidence and do not let unrelated successful "
        "placements of other objects dominate your choice. "
        "If embodied_memorizer Retrieved Events are present, use those event records as primary evidence; "
        "do not let unrelated spatial memories override failed/corrected placement evidence. "
        "If embodied_memorizer Latest Placement Facts are present, treat them as the current object-location "
        "facts with highest priority. "
        "Pickupable objects may have Navigate actions; use Navigate before PickUp when needed. "
        "Do not choose PutObject when held_object is null/nothing; pick up the object first. "
        "Only choose Done after the task objective is satisfied. For placement or storage "
        "instructions, a successful PickUp is only an intermediate step; continue until the "
        "object is placed in the inferred target location. For retrieval-only instructions, "
        "Done after pickup can be appropriate. "
        "For hide/store instructions involving an openable container, after a successful "
        "PutObject prefer Close on that same container before Done when Close is available. "
        "PickUp is exposed only when the object is expected to be visible/reachable; if pickup fails, "
        "change the state first by navigating to a nearby receptacle/surface, moving, "
        "rotating, or opening a container instead of repeating it. "
        "If a Navigate, Open, Close, or TransferContents action just succeeded, do not repeat "
        "the same action on the same target; choose the next useful action such as PickUp, "
        "PutObject, opening a different container, or Done when the objective is satisfied.\n"
        + compact_legend
        + "\n"
        + json.dumps(
            serialized_payload,
            ensure_ascii=False,
            indent=None if compact_json else 2,
            separators=(",", ":") if compact_json else None,
        )
    )


def build_sequence_user_prompt(
    *,
    obs: dict,
    history: list[dict],
    max_context_steps: int,
    max_visible_objects: int,
    max_actions: int,
    compact_json: bool = False,
) -> str:
    payload = {
        "instruction": obs.get("instruction"),
        "held_object": compact_held_object(obs),
        "memory_context": compact_context(
            obs.get("context", []),
            max_context_steps,
            obs.get("object_labels", {}),
        ),
        "current_visible_objects": compact_visible_objects(obs, max_visible_objects),
        "previous_model_steps": compact_history(history),
        "initial_available_actions": compact_available_actions(
            obs.get("available_actions", [])
        ),
        "max_actions": max_actions,
    }
    serialized_payload = _compact_prompt_payload(payload) if compact_json else payload
    return (
        "Plan the full next action sequence for this probe in one response. "
        "Return at most max_actions actions. Use the initial_available_actions for the first "
        "step and semantic action fields for later steps because future action_id values may "
        "change after execution. Prefer this action object schema: "
        '{"action_type": "Navigate", "target_label": "Fridge", "target_type": "Fridge"}. '
        'If an action has no target, such as Done, use {"action_type": "Done"}. '
        "If previous_model_steps contains a failed action or failed plan, explain why it failed "
        "and output a revised sequence from the current state; do not repeat the same failed "
        "action unless a successful action has changed the state. "
        "Do not include hidden oracle fields or raw simulator object ids.\n"
        + json.dumps(
            serialized_payload,
            ensure_ascii=False,
            indent=None if compact_json else 2,
            separators=(",", ":") if compact_json else None,
        )
    )


def build_messages(
    *,
    obs: dict,
    img_path: str,
    history: list[dict],
    text_only: bool,
    max_context_steps: int,
    max_visible_objects: int,
    compact_json: bool = False,
) -> list[dict]:
    text = build_user_prompt(
        obs=obs,
        history=history,
        max_context_steps=max_context_steps,
        max_visible_objects=max_visible_objects,
        compact_json=compact_json,
    )
    if text_only:
        user_content: str | list[dict] = text
    else:
        user_content = [{"type": "text", "text": text}]
        historical_images = list(obs.get("historical_images") or [])
        for image_index, historical in enumerate(historical_images, start=1):
            user_content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            "Historical RGB frame "
                            f"{image_index}/{len(historical_images)} "
                            f"(session {historical.get('session_index')}, "
                            f"step {historical.get('step_index')}; chronological order)."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(str(historical["path"])),
                            "detail": "high",
                        },
                    },
                ]
            )
        user_content.extend(
            [
                {"type": "text", "text": "Current 500x500 RGB observation."},
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(img_path)},
                },
            ]
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_sequence_messages(
    *,
    obs: dict,
    img_path: str,
    history: list[dict],
    text_only: bool,
    max_context_steps: int,
    max_visible_objects: int,
    max_actions: int,
    compact_json: bool = False,
) -> list[dict]:
    text = build_sequence_user_prompt(
        obs=obs,
        history=history,
        max_context_steps=max_context_steps,
        max_visible_objects=max_visible_objects,
        max_actions=max_actions,
        compact_json=compact_json,
    )
    if text_only:
        user_content: str | list[dict] = text
    else:
        user_content = [{"type": "text", "text": text}]
        historical_images = list(obs.get("historical_images") or [])
        for image_index, historical in enumerate(historical_images, start=1):
            user_content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            "Historical RGB frame "
                            f"{image_index}/{len(historical_images)} "
                            f"(session {historical.get('session_index')}, "
                            f"step {historical.get('step_index')}; chronological order)."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(str(historical["path"])),
                            "detail": "high",
                        },
                    },
                ]
            )
        user_content.extend(
            [
                {"type": "text", "text": "Current 500x500 RGB observation."},
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(img_path)},
                },
            ]
        )
    return [
        {"role": "system", "content": SEQUENCE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def extract_json_object(text: str) -> dict[str, Any]:
    def loads_with_trailing_brace_repair(candidate: str) -> dict[str, Any]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            missing_closers = candidate.count("{") - candidate.count("}")
            if missing_closers > 0 and candidate.rstrip().endswith("}"):
                return json.loads(candidate + ("}" * missing_closers))
            raise

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return loads_with_trailing_brace_repair(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        return loads_with_trailing_brace_repair(match.group(0))


def parse_model_action(content: str) -> tuple[Any, str, dict]:
    try:
        payload = extract_json_object(content)
    except json.JSONDecodeError:
        # Some OpenAI-compatible Gemini endpoints occasionally truncate the
        # free-text reasoning while already emitting a clear action_id. Recover
        # only that explicit id; never infer a target or action semantics.
        match = re.search(r'"action_id"\s*:\s*(-?\d+)', content)
        if not match:
            raise
        payload = {
            "action_id": int(match.group(1)),
            "reasoning": "",
            "recovered_from_partial_json": True,
        }
    reasoning = payload.get("reasoning") or payload.get("thought") or ""

    if "action_id" in payload:
        return int(payload["action_id"]), reasoning, payload
    if isinstance(payload.get("action"), dict):
        return payload["action"], reasoning, payload
    if isinstance(payload.get("actions"), list) and payload["actions"]:
        first = payload["actions"][0]
        if isinstance(first, int):
            return first, reasoning, payload
        if isinstance(first, dict) and "action_id" in first:
            return int(first["action_id"]), first.get("reasoning", reasoning), payload
        return first, reasoning, payload
    if (
        isinstance(payload.get("selected_action_plan"), list)
        and payload["selected_action_plan"]
    ):
        first = payload["selected_action_plan"][0]
        if isinstance(first, dict):
            if "action_id" in first:
                return (
                    int(first["action_id"]),
                    first.get("reasoning", reasoning),
                    payload,
                )
            if first.get("action_type"):
                return first, reasoning, payload
        if isinstance(first, int):
            return first, reasoning, payload
    if payload.get("action_type"):
        return payload, reasoning, payload

    raise ValueError(f"No action found in model response: {content[:200]}")


def parse_model_action_sequence(
    content: str, max_actions: int
) -> tuple[list[Any], str, dict]:
    payload = extract_json_object(content)
    reasoning = payload.get("reasoning") or payload.get("thought") or ""
    actions = (
        payload.get("actions")
        or payload.get("action_sequence")
        or payload.get("selected_action_plan")
        or payload.get("plan")
    )
    if actions is None and "action_id" in payload:
        actions = [{"action_id": payload["action_id"]}]
    if actions is None and payload.get("action_type"):
        actions = [payload]
    if not isinstance(actions, list):
        raise ValueError(f"No action sequence found in model response: {content[:200]}")

    parsed: list[Any] = []
    for item in actions[:max_actions]:
        if isinstance(item, int):
            parsed.append({"action_id": item})
            continue
        if isinstance(item, dict):
            if "action_id" in item and "action_type" not in item:
                parsed.append({"action_id": item["action_id"]})
            else:
                parsed.append(canonicalize_sequence_action(item))
            continue
        if isinstance(item, str) and item.strip().lower() in {"done", "stop"}:
            parsed.append({"action_type": "Done"})
    if not parsed:
        raise ValueError(f"Empty action sequence in model response: {content[:200]}")
    for key in ("actions", "action_sequence", "selected_action_plan", "plan"):
        if isinstance(payload.get(key), list):
            payload = dict(payload)
            payload[key] = parsed
            break
    return parsed, reasoning, payload


class APIPlanner:
    """Stateful planner wrapper around a chat-completions compatible API."""

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str | None,
        api_key: str | None,
        max_tokens: int,
        temperature: float,
        timeout: int,
        retries: int,
        retry_delay: float,
        text_only: bool,
        max_context_steps: int,
        max_visible_objects: int,
        compact_json_prompt: bool = False,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay
        self.text_only = text_only
        self.max_context_steps = max_context_steps
        self.max_visible_objects = max_visible_objects
        self.compact_json_prompt = compact_json_prompt
        self.history: list[dict] = []
        self.model_calls: list[dict[str, Any]] = []
        self.planner_steps = 0
        self.output_json_error = 0

    def reset(self):
        self.history = []
        self.model_calls = []
        self.planner_steps = 0
        self.output_json_error = 0

    def _record_model_call(
        self,
        *,
        messages: list[dict],
        content: str,
        metadata: dict[str, Any],
        phase: str,
    ) -> None:
        request_text = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        prompt_estimate = max(1, (len(request_text) + 3) // 4)
        completion_estimate = max(1, (len(content) + 3) // 4)
        row = dict(metadata)
        row["prompt_tokens"] = int(row.get("prompt_tokens") or prompt_estimate)
        row["completion_tokens"] = int(
            row.get("completion_tokens") or completion_estimate
        )
        row["total_tokens"] = int(
            row.get("total_tokens") or row["prompt_tokens"] + row["completion_tokens"]
        )
        row["usage_source"] = (
            "server"
            if metadata.get("prompt_tokens") is not None
            and metadata.get("completion_tokens") is not None
            else "approx_chars_div4"
        )
        row["request_bytes"] = len(request_text.encode("utf-8"))
        row["response_bytes"] = len(content.encode("utf-8"))
        row["phase"] = phase
        self.model_calls.append(row)

    def act(self, obs: dict) -> tuple[Any, str, dict, str]:
        img_path = obs["image_path"]
        messages = build_messages(
            obs=obs,
            img_path=img_path,
            history=self.history,
            text_only=self.text_only,
            max_context_steps=self.max_context_steps,
            max_visible_objects=self.max_visible_objects,
            compact_json=self.compact_json_prompt,
        )
        self.planner_steps += 1
        content = ""
        metadata: dict[str, Any] = {}
        try:
            content = post_chat_completion(
                model_name=self.model_name,
                base_url=self.base_url,
                api_key=self.api_key,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
                retries=self.retries,
                retry_delay=self.retry_delay,
                metadata_sink=metadata,
            )
            self._record_model_call(
                messages=messages,
                content=content,
                metadata=metadata,
                phase="action",
            )
            action, reasoning, payload = parse_model_action(content)
            return action, reasoning, payload, content
        except Exception as exc:
            if isinstance(exc, APIBalanceError):
                raise
            if isinstance(exc, APITransportError):
                raise
            if is_loopback_service_unavailable(base_url=self.base_url, error=exc):
                raise LocalAPIServiceUnavailableError(str(exc)) from exc
            self.output_json_error += 1
            return (
                -1,
                f"API/parse error: {exc}",
                {"error": str(exc), "raw_text": content},
                content,
            )

    def plan_sequence(
        self, obs: dict, *, max_actions: int
    ) -> tuple[list[Any], str, dict, str]:
        img_path = obs["image_path"]
        messages = build_sequence_messages(
            obs=obs,
            img_path=img_path,
            history=self.history,
            text_only=self.text_only,
            max_context_steps=self.max_context_steps,
            max_visible_objects=self.max_visible_objects,
            max_actions=max_actions,
            compact_json=self.compact_json_prompt,
        )
        self.planner_steps += 1
        content = ""
        metadata: dict[str, Any] = {}
        try:
            content = post_chat_completion(
                model_name=self.model_name,
                base_url=self.base_url,
                api_key=self.api_key,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout,
                retries=self.retries,
                retry_delay=self.retry_delay,
                metadata_sink=metadata,
            )
            self._record_model_call(
                messages=messages,
                content=content,
                metadata=metadata,
                phase="sequence_action",
            )
            actions, reasoning, payload = parse_model_action_sequence(
                content, max_actions
            )
            return actions, reasoning, payload, content
        except Exception as exc:
            if isinstance(exc, APIBalanceError):
                raise
            if isinstance(exc, APITransportError):
                raise
            if is_loopback_service_unavailable(base_url=self.base_url, error=exc):
                raise LocalAPIServiceUnavailableError(str(exc)) from exc
            self.output_json_error += 1
            return (
                [],
                f"API/parse error: {exc}",
                {"error": str(exc), "raw_text": content},
                content,
            )

    def update_info(self, info: dict, action, raw_payload: dict, raw_text: str) -> dict:
        step_record = {
            "step": info.get("env_step"),
            "action": action,
            "success": info.get("last_action_success", 0.0) > 0,
            "feedback": info.get("env_feedback"),
            "model_response": raw_payload,
            "raw_text": raw_text,
        }
        self.history.append(step_record)
        return step_record
