"""Retained protocol helpers from the original experiment code."""

from __future__ import annotations


import re


def short_error(exc: Exception, limit: int = 1200) -> str:
    text = f"{type(exc).__name__}: {exc}"
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _semantic_key(value) -> str:
    text = str(value or "").strip()
    if "|" in text:
        text = text.split("|", 1)[0]
    text = re.sub(r"_\d+$", "", text)
    text = re.sub(r"[^A-Za-z0-9]+", "", text).lower()
    if len(text) > 3 and text.endswith("ies"):
        text = text[:-3] + "y"
    elif len(text) > 3 and text.endswith("es"):
        text = text[:-2]
    elif len(text) > 3 and text.endswith("s"):
        text = text[:-1]
    return text


def _instance_semantic_key(value) -> str:
    """Normalize a target label while preserving an instance suffix such as `_2`."""
    text = str(value or "").strip()
    if "|" in text:
        text = text.split("|", 1)[0]
    text = re.sub(r"___\d+$", "", text)
    return re.sub(r"[^A-Za-z0-9]+", "", text).lower()


def _specific_target_keys_match(cue: str, candidate: str) -> bool:
    if not cue or not candidate:
        return False
    if cue == candidate:
        return True
    cue_is_instance = cue[-1:].isdigit()
    candidate_is_instance = candidate[-1:].isdigit()
    if cue_is_instance:
        return False
    if candidate_is_instance:
        return candidate.startswith(cue)
    return cue in candidate or candidate in cue


def _navigate_target_keys_match(cue: str, candidate: str) -> bool:
    if not cue or not candidate:
        return False
    if cue == candidate:
        return True
    if cue[-1:].isdigit():
        return False
    return candidate[-1:].isdigit() and candidate.startswith(cue)


def _target_slot_base(value) -> str:
    text = str(value or "").strip()
    return re.sub(r"___\d+$", "", text)


def _reasoning_action_type_and_targets(reasoning: object) -> tuple[str, set[str]]:
    text = str(reasoning or "")
    if not text:
        return "", set()
    action_type = ""
    for candidate, pattern in [
        ("Navigate", r"\bNavigate\b"),
        ("PickUp", r"\b(?:PickUp|Pick up)\b"),
        ("PutObject", r"\b(?:PutObject|Put Object|Put held object|Place)\b"),
        ("Open", r"\bOpen\b"),
        ("Close", r"\bClose\b"),
        ("Done", r"\b(?:Done|Stop)\b"),
    ]:
        if re.search(pattern, text, re.I):
            action_type = candidate
            break
    target_cues: set[str] = set()
    for pattern in [
        r"\b(?:for|to|on/in|into|in|on)\s+(?:the\s+)?(?P<target>[A-Za-z][A-Za-z0-9_]*)",
        r"\b(?:Open|Navigate to|Pick up|Close)\s+(?:the\s+)?(?P<target>[A-Za-z][A-Za-z0-9_]*)",
    ]:
        for match in re.finditer(pattern, text, re.I):
            cue = _semantic_key(match.group("target"))
            if cue and cue not in {"current", "next", "oracle", "target", "step"}:
                target_cues.add(cue)
    return action_type, target_cues


def repair_action_id_from_structured_prediction(
    action,
    raw_payload: dict,
    obs: dict,
) -> tuple[object, dict | None]:
    """Repair stale action_id when the model's structured fields disagree.

    MemoryActionSFT-v2 sometimes predicts the right semantic next action in
    `supervised_action_type` / `supervised_target_type` but emits a stale
    action_id from the previous step. This repair uses only model-visible
    fields and the current available action list; it never consults oracle
    targets or evaluator state.
    """
    if not isinstance(raw_payload, dict):
        return action, None
    predicted_type = raw_payload.get("supervised_action_type")
    if not predicted_type and isinstance(raw_payload.get("action"), dict):
        predicted_type = raw_payload["action"].get("action_type")
    reasoning_type, reasoning_target_cues = _reasoning_action_type_and_targets(
        raw_payload.get("reasoning") or raw_payload.get("thought")
    )
    if not predicted_type:
        predicted_type = reasoning_type
    predicted_type = str(predicted_type or "").strip()
    if not predicted_type:
        return action, None

    actions = list(obs.get("available_actions") or [])
    if not actions:
        return action, None

    by_action_id = {
        item.get("action_id"): item
        for item in actions
        if isinstance(item.get("action_id"), int)
    }
    original_entry = None
    if isinstance(action, int):
        original_entry = by_action_id.get(action)

    candidates = [item for item in actions if item.get("action_type") == predicted_type]
    fallback_action_type = None
    if not candidates and predicted_type == "PickUp":
        fallback_action_type = "Navigate"
        candidates = [
            item for item in actions if item.get("action_type") == fallback_action_type
        ]
    if not candidates and predicted_type == "Open":
        fallback_action_type = "Navigate"
        candidates = [
            item for item in actions if item.get("action_type") == fallback_action_type
        ]
    if not candidates:
        return action, None

    target_cues = {
        _semantic_key(raw_payload.get("supervised_target_type")),
        _semantic_key(raw_payload.get("supervised_target_label")),
        _semantic_key(raw_payload.get("target_type")),
        _semantic_key(raw_payload.get("target_label")),
    }
    target_cues.update(reasoning_target_cues)
    if isinstance(raw_payload.get("action"), dict):
        target_cues.update(
            {
                _semantic_key(raw_payload["action"].get("target_type")),
                _semantic_key(raw_payload["action"].get("target_label")),
                _semantic_key(raw_payload["action"].get("target")),
            }
        )
    target_cues.discard("")

    original_target = original_entry.get("target") if original_entry else None
    original_label = original_entry.get("target_label") if original_entry else None
    original_type = original_entry.get("target_type") if original_entry else None
    original_label_key = _semantic_key(original_label)
    original_type_key = _semantic_key(original_type)

    def entry_matches_target_cues(entry: dict | None) -> bool:
        if entry is None or not target_cues:
            return True
        entry_keys = {
            _semantic_key(entry.get("target")),
            _semantic_key(entry.get("target_label")),
            _semantic_key(entry.get("target_type")),
        }
        entry_keys.discard("")
        for key in entry_keys:
            if key in target_cues:
                return True
            if any(cue and (cue in key or key in cue) for cue in target_cues):
                return True
        return False

    original_matches_target_cues = entry_matches_target_cues(original_entry)
    if (
        original_entry is not None
        and original_entry.get("action_type") == predicted_type
        and original_matches_target_cues
    ):
        return action, None

    def score(candidate: dict) -> tuple[int, int]:
        candidate_label_key = _semantic_key(candidate.get("target_label"))
        candidate_type_key = _semantic_key(candidate.get("target_type"))
        candidate_target = candidate.get("target")
        value = 0
        if original_entry and original_matches_target_cues:
            if candidate_target and candidate_target == original_target:
                value += 160
            if (
                candidate_target
                and original_target
                and _target_slot_base(candidate_target)
                == _target_slot_base(original_target)
            ):
                value += 150
            if candidate_label_key and candidate_label_key == original_label_key:
                value += 150
            if candidate_type_key and candidate_type_key == original_type_key:
                value += 40
        if target_cues:
            if candidate_label_key in target_cues:
                value += 120
            if candidate_type_key in target_cues:
                value += 90
            for cue in target_cues:
                if cue and (
                    cue in candidate_label_key
                    or cue in candidate_type_key
                    or candidate_label_key in cue
                ):
                    value += 20
                    break
        if not candidate.get("target") and predicted_type in {"Done", "Stop"}:
            value += 50
        action_id = int(candidate.get("action_id", -1))
        return value, -action_id

    best = max(candidates, key=score)
    best_score, _ = score(best)
    if best_score <= 0 and predicted_type == "Open" and fallback_action_type is None:
        fallback_candidates = [
            item for item in actions if item.get("action_type") == "Navigate"
        ]
        if fallback_candidates:
            fallback_action_type = "Navigate"
            candidates = fallback_candidates
            best = max(candidates, key=score)
            best_score, _ = score(best)
    if best_score <= 0:
        return action, None

    corrected_action = int(best.get("action_id", -1))
    if corrected_action < 0 or corrected_action == action:
        return action, None
    correction = {
        "original_action_id": action if isinstance(action, int) else None,
        "corrected_action_id": corrected_action,
        "predicted_action_type": predicted_type,
        "predicted_target_type": raw_payload.get("supervised_target_type"),
        "original_action_type": original_entry.get("action_type")
        if original_entry
        else None,
        "original_target_label": original_label,
        "corrected_action_type": best.get("action_type"),
        "corrected_target_label": best.get("target_label"),
        "corrected_target_type": best.get("target_type"),
        "fallback_action_type": fallback_action_type,
        "repair_score": best_score,
    }
    return corrected_action, correction


def bind_sequence_action_to_observation_space(
    action, obs: dict
) -> tuple[object, dict | None]:
    """Ground a semantic sequence action to the current available action list.

    In sequence mode the model plans future actions before the environment has
    assigned future action ids.  This binds action_type/target labels against
    the current observation just before each step is executed.
    """
    if not isinstance(action, dict):
        return action, None

    action_type = action.get("action_type") or action.get("action")
    if not action_type:
        return action, None
    if action_type == "Stop":
        action_type = "Done"
    if action_type == "Done":
        return {"action_type": "Done"}, None

    planned_id = None
    if "action_id" in action:
        try:
            planned_id = int(action["action_id"])
        except (TypeError, ValueError):
            planned_id = None

    specific_target_cues = {
        _instance_semantic_key(action.get("target")),
        _instance_semantic_key(action.get("objectId")),
        _instance_semantic_key(action.get("target_label")),
        _instance_semantic_key(action.get("parent_target_label")),
    }
    specific_target_cues.discard("")
    type_target_cues = {
        _semantic_key(action.get("target_type")),
        _semantic_key(action.get("parent_target_type")),
    }
    type_target_cues.discard("")
    target_cues = specific_target_cues | type_target_cues

    def score(candidate: dict) -> tuple[int, int]:
        if action_type == "Navigate" and candidate.get("target_label"):
            candidate_direct_specific_keys = {
                _instance_semantic_key(candidate.get("target_label")),
            }
        else:
            candidate_direct_specific_keys = {
                _instance_semantic_key(candidate.get("target")),
                _instance_semantic_key(candidate.get("target_label")),
            }
        candidate_direct_specific_keys.discard("")
        candidate_parent_specific_keys = {
            _instance_semantic_key(candidate.get("parent_target")),
            _instance_semantic_key(candidate.get("parent_target_label")),
        }
        candidate_parent_specific_keys.discard("")
        candidate_direct_type_keys = {
            _semantic_key(candidate.get("target_type")),
        }
        candidate_direct_type_keys.discard("")
        candidate_parent_type_keys = {
            _semantic_key(candidate.get("parent_target_type")),
        }
        candidate_parent_type_keys.discard("")
        direct_specific_score = sum(
            300 if cue == key else 60
            for cue in specific_target_cues
            for key in candidate_direct_specific_keys
            if (
                _navigate_target_keys_match(cue, key)
                if action_type == "Navigate"
                else _specific_target_keys_match(cue, key)
            )
        )
        parent_specific_score = 0
        if action_type != "Navigate":
            parent_specific_score = sum(
                80 if cue == key else 15
                for cue in specific_target_cues
                for key in candidate_parent_specific_keys
                if _specific_target_keys_match(cue, key)
            )
        direct_type_score = sum(
            30 if cue == key else 8
            for cue in type_target_cues
            for key in candidate_direct_type_keys
            if cue == key or cue in key or key in cue
        )
        parent_type_score = sum(
            6 if cue == key else 2
            for cue in type_target_cues
            for key in candidate_parent_type_keys
            if cue == key or cue in key or key in cue
        )
        specific_score = direct_specific_score + parent_specific_score
        value = specific_score + direct_type_score + parent_type_score
        if specific_target_cues and specific_score <= 0:
            value = 0
        elif planned_id is not None and candidate.get("action_id") == planned_id:
            value += 40
        try:
            action_id = int(candidate.get("action_id", -1))
        except (TypeError, ValueError):
            action_id = -1
        return value, -action_id

    actions = [
        item
        for item in (obs.get("available_actions") or [])
        if item.get("action_type") == action_type
    ]
    if not actions:
        return -1, {
            "reason": "action_type_not_in_current_observation_space",
            "planned_action_type": action_type,
        }

    by_action_id = {
        item.get("action_id"): item
        for item in actions
        if isinstance(item.get("action_id"), int)
    }

    if not target_cues:
        if planned_id in by_action_id:
            return dict(by_action_id[planned_id]), {
                "planned_action": action,
                "bound_action_id": planned_id,
                "reason": "matched_planned_action_id",
            }
        if len(actions) == 1:
            only = dict(actions[0])
            return only, {
                "planned_action": action,
                "bound_action_id": only.get("action_id"),
                "reason": "only_candidate_for_action_type",
            }
        return -1, {
            "reason": "ambiguous_targetless_sequence_action",
            "planned_action_type": action_type,
            "candidate_count": len(actions),
        }

    best = max(actions, key=score)
    best_score, _ = score(best)
    if best_score <= 0:
        return -1, {
            "reason": "no_target_match_for_sequence_action",
            "planned_action_type": action_type,
            "target_cues": sorted(target_cues),
            "candidate_count": len(actions),
        }

    bound = dict(best)
    return bound, {
        "planned_action": action,
        "bound_action_id": bound.get("action_id"),
        "bound_action_type": bound.get("action_type"),
        "bound_target_label": bound.get("target_label"),
        "bound_target_type": bound.get("target_type"),
        "binding_score": best_score,
    }


def bind_action_to_observation_space(action, obs: dict) -> tuple[object, dict | None]:
    """Bind integer action ids to the exact action entry visible to the model.

    embodied_memorizer may filter the observation action list.  The environment keeps
    its full internal action list, so executing a bare integer can accidentally
    select an action that was not present in the model-visible observation.
    """
    actions = list(obs.get("available_actions") or [])
    if not actions:
        return action, None
    if isinstance(action, dict) and action.get("action_type"):
        return action, None
    try:
        action_id = int(action["action_id"] if isinstance(action, dict) else action)
    except (TypeError, ValueError, KeyError):
        return action, None
    for item in actions:
        if item.get("action_id") == action_id:
            return dict(item), None
    return -1, {
        "invalid_action_id": action_id,
        "reason": "action_id_not_in_current_observation_space",
    }
