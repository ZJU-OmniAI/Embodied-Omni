"""
Experience memory: consolidate raw event streams into reusable embodied rules.

The event layer stores what happened. This layer stores what was learned from
what happened, such as "putting X on Y failed, then a human corrected it to Z".
It is deliberately separate from prompt construction so storage, consolidation,
and retrieval can be improved without rewriting downstream prompts.
"""

import re
from typing import List, Optional


PLACEMENT_RE = re.compile(
    r"\b(?:Place|Put|Try placing|Move)\s+"
    r"(?:(?:cleaned|cooled|heated|sliced|washed|filled|empty|dirty|cooked)\s+)?"
    r"(?P<object>[A-Za-z][A-Za-z0-9_]*)\s+"
    r"(?:casually\s+)?(?:in|into|on|onto|to)\s+"
    r"(?:the\s+)?(?P<target>[A-Za-z][A-Za-z0-9_]*)",
    re.I,
)
NAVIGATE_RE = re.compile(
    r"\bNavigate\s+to\s+(?:the\s+)?(?P<target>[A-Za-z][A-Za-z0-9_]*)",
    re.I,
)
PICKUP_RE = re.compile(
    r"\bPick\s+up\s+(?:the\s+)?(?P<object>[A-Za-z][A-Za-z0-9_]*)",
    re.I,
)
INTERVENTION_TARGET_RE = re.compile(
    r"\bHuman\s+intervention:\s+"
    r"(?:the\s+)?(?P<object>[A-Za-z][A-Za-z0-9_]*)\s+"
    r"(?:should|must|needs?\s+to)\s+"
    r"(?:be\s+)?(?:returned|placed|moved|kept|stored|put)\s+"
    r"(?:back\s+)?(?:in|into|on|onto|to)\s+"
    r"(?:the\s+)?(?P<target>[A-Za-z][A-Za-z0-9_]*)",
    re.I,
)
OBSERVED_LOCATION_RE = re.compile(
    r"\b(?P<object>[A-Za-z][A-Za-z0-9_]*)\s+"
    r"(?:was\s+)?(?:seen|observed|located|left|placed|stored)\s+"
    r"(?:currently\s+)?(?P<relation>on/in|in|inside|into|on|onto|at)\s+"
    r"(?:the\s+)?(?P<target>[A-Za-z][A-Za-z0-9_]*)",
    re.I,
)
TARGET_ANNOTATION_RE = re.compile(
    r"\[target:\s*(?P<target>[A-Za-z][A-Za-z0-9_]*)\]",
    re.I,
)
INTERACTION_ACTION_PATTERNS = [
    (
        "Open",
        re.compile(
            r"(?:\bOpen\s+(?:the\s+)?|打开\s*)(?P<target>[A-Za-z][A-Za-z0-9_]*)", re.I
        ),
    ),
    (
        "Close",
        re.compile(
            r"(?:\bClose\s+(?:the\s+)?|关闭\s*)(?P<target>[A-Za-z][A-Za-z0-9_]*)", re.I
        ),
    ),
    (
        "PickUp",
        re.compile(r"\bPick\s+up\s+(?:the\s+)?(?P<target>[A-Za-z][A-Za-z0-9_]*)", re.I),
    ),
    (
        "Navigate",
        re.compile(
            r"(?:\bNavigate\s+to\s+(?:the\s+)?|走向\s*)(?P<target>[A-Za-z][A-Za-z0-9_]*)",
            re.I,
        ),
    ),
    (
        "PutObject",
        re.compile(
            r"\b(?:Put|Place|Try placing|Move)\s+"
            r"(?:[A-Za-z][A-Za-z0-9_]*\s+)?"
            r"(?:in|into|on|onto|to)\s+(?:the\s+)?(?P<target>[A-Za-z][A-Za-z0-9_]*)",
            re.I,
        ),
    ),
]


def _singular(text: str) -> str:
    lowered = str(text or "").lower()
    if len(lowered) > 3 and lowered.endswith("ies"):
        return lowered[:-3] + "y"
    if len(lowered) > 3 and lowered.endswith("es"):
        return lowered[:-2]
    if len(lowered) > 3 and lowered.endswith("s"):
        return lowered[:-1]
    return lowered


def _base_label(text: str) -> str:
    label = str(text or "").strip()
    if "|" in label:
        label = label.split("|", 1)[0]
    label = re.sub(r"_\d+$", "", label)
    return label


def _instance_label(text: str) -> str:
    label = str(text or "").strip()
    if "|" in label:
        label = label.split("|", 1)[0]
    return label


def _canonical_label(text: str) -> str:
    return _singular(_base_label(text))


def _instance_key(text: str) -> str:
    return _singular(_instance_label(text))


def _identity_key(text: str) -> str:
    value = str(text or "").strip()
    if "|" not in value:
        return _instance_key(value)
    value = value.lower()
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def _terms(text: str) -> set[str]:
    out: set[str] = set()
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9_]*", str(text or "")):
        pieces = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", raw)
        candidates = [raw, *pieces]
        if len(pieces) > 1:
            candidates.append(" ".join(pieces))
        for term in candidates:
            norm = _singular(term)
            if len(norm) >= 3 or (len(norm) >= 2 and term.isupper()):
                out.add(norm)
    return out


def _semantic_category_key(term: str) -> str:
    term = _singular(term)
    return OBJECT_CATEGORY_ALIASES.get(term, term)


def _raw_label_terms(text: str) -> set[str]:
    return {
        _singular(raw)
        for raw in re.findall(r"[A-Za-z][A-Za-z0-9_]*", str(text or ""))
        if raw
    }


def _object_categories(text: str) -> set[str]:
    """Infer semantic categories while avoiding noisy component over-expansion."""
    primary_terms = {
        _canonical_label(text),
        _instance_key(text),
        *_raw_label_terms(text),
    }
    categories: set[str] = set()
    for term in primary_terms:
        category_key = _semantic_category_key(term)
        if category_key in OBJECT_SEMANTIC_CATEGORIES:
            categories.add(category_key)
        categories.update(MEMBER_TO_OBJECT_CATEGORIES.get(category_key, set()))
    if categories:
        return categories

    for term in _terms(text):
        category_key = _semantic_category_key(term)
        if category_key in OBJECT_SEMANTIC_CATEGORIES:
            categories.add(category_key)
        categories.update(MEMBER_TO_OBJECT_CATEGORIES.get(category_key, set()))
    return categories


def _object_semantic_terms(text: str) -> set[str]:
    """Return exact lexical terms plus learned-generalization category terms."""
    return _terms(text) | _object_categories(text)


def _expanded_object_query_terms(query: str) -> tuple[set[str], set[str]]:
    """Split object query terms into exact terms and lower-priority semantic terms."""
    exact = _terms(query)
    expanded = set(exact)
    expanded.update(_object_categories(query))
    for term in list(exact):
        category_key = _semantic_category_key(term)
        if category_key in OBJECT_SEMANTIC_CATEGORIES:
            expanded.add(category_key)
            expanded.update(OBJECT_SEMANTIC_CATEGORIES[category_key])
        for category in MEMBER_TO_OBJECT_CATEGORIES.get(category_key, set()):
            expanded.add(category)
    return exact, expanded - exact


def _placement_relation(text: str) -> str:
    if re.search(r"\bon/in\b", text, re.I):
        return "on_or_in"
    if re.search(r"\b(?:in|into|inside)\b", text, re.I):
        return "in"
    return "on"


HABIT_QUERY_CUES = {
    "habit",
    "household",
    "implicit",
    "preference",
    "preferred",
    "should",
    "store",
    "storage",
    "placement",
    "place",
    "belong",
    "belongs",
    "correct",
}

LOCATION_QUERY_CUES = {
    "where",
    "find",
    "bring",
    "retrieve",
    "get",
    "latest",
    "current",
    "currently",
    "seen",
    "observed",
    "located",
    "left",
}

LOCATION_QUERY_SUBSTRINGS = (
    "拿过来",
    "拿来",
    "取回",
    "取来",
    "带过来",
    "带来",
    "找",
    "在哪里",
    "在哪",
)

CONSTRAINT_QUERY_CUES = {
    "avoid",
    "can",
    "candidate",
    "failure",
    "failed",
    "locked",
    "openable",
    "unreachable",
    "invalid",
    "cannot",
    "container",
    "usable",
    "openable",
    "hide",
    "hidden",
    "mistake",
}

CONSTRAINT_QUERY_SUBSTRINGS = (
    "失败",
    "锁",
    "打不开",
    "不能打开",
    "避开",
    "藏",
    "隐蔽",
    "收纳容器",
    "容器",
    "打开",
)

DURABLE_FAILURE_MODES = {"locked", "invalid_placement"}
DURABLE_SUCCESS_ACTIONS = {"Open"}

BLOCKED_ACTION_TYPES_BY_FAILURE = {
    ("Open", "locked"): {"Open", "PutObject", "TransferContents"},
    ("PutObject", "invalid_placement"): {
        "Navigate",
        "Open",
        "Close",
        "PutObject",
        "TransferContents",
    },
}

SOURCE_WEIGHTS = {
    "successful_relocation": 1.05,
    "successful_placement": 1.0,
    "explicit_memory_cue": 1.2,
    "successful_pickup": 0.9,
    "post_intervention_placement": 0.95,
    "post_intervention_attempt": 0.85,
    "post_intervention_followup_placement": 0.95,
    "post_intervention_navigation_attempt": 0.65,
    "explicit_human_intervention": 0.9,
    "observed_location_event": 0.8,
    "interaction_failure_event": 0.85,
    "interaction_success_event": 0.8,
}

HABIT_SOURCE_AUTHORITY = {
    # Completed or explicit human-corrected placements are durable policy
    # evidence.  We keep these above failed attempts so weak late signals do
    # not erase a stronger household rule.
    "post_intervention_placement": 3,
    "post_intervention_followup_placement": 3,
    "explicit_human_intervention": 3,
    # The intended corrected target is useful, but the physical action did not
    # complete; use it only when no stronger correction exists.
    "post_intervention_attempt": 2,
    # Navigation after intervention is the weakest target cue.
    "post_intervention_navigation_attempt": 1,
}

# Lightweight embodied-object taxonomy used by the experience index.
#
# This deliberately lives in the embodied_memorizer storage/retrieval layer rather
# than in prompts.  It lets a habit observed for Plate/Bowl support a later
# "dishware" query, while exact object matches still dominate category matches
# during ranking.
OBJECT_SEMANTIC_CATEGORIES = {
    "beverage": {
        "bottle",
        "cup",
        "glass",
        "mug",
        "waterbottle",
        "winebottle",
        "wineglass",
    },
    "cleaning": {
        "cloth",
        "dishsponge",
        "handtowel",
        "papertowel",
        "soapbar",
        "soapbottle",
        "sponge",
        "spraybottle",
        "tissuebox",
        "toiletpaper",
    },
    "container": {
        "basket",
        "bottle",
        "bowl",
        "box",
        "cup",
        "drawer",
        "mug",
        "pot",
        "vase",
    },
    "cookware": {
        "bowl",
        "kettle",
        "pan",
        "plate",
        "pot",
    },
    "dishware": {
        "bowl",
        "cup",
        "dish",
        "glass",
        "mug",
        "plate",
        "wineglass",
    },
    "decorative": {
        "candle",
        "houseplant",
        "painting",
        "statue",
        "tabletopdecor",
        "vase",
    },
    "document": {
        "book",
        "cd",
        "creditcard",
        "newspaper",
        "notebook",
        "paper",
        "pen",
        "pencil",
    },
    "electronic": {
        "alarmclock",
        "cellphone",
        "cd",
        "laptop",
        "remotecontrol",
        "tablet",
        "television",
        "watch",
    },
    "food": {
        "apple",
        "bread",
        "egg",
        "lettuce",
        "potato",
        "tomato",
    },
    "personal": {
        "alarmclock",
        "book",
        "cellphone",
        "creditcard",
        "keychain",
        "laptop",
        "pen",
        "pencil",
        "watch",
    },
    "toiletry": {
        "handtowel",
        "soapbar",
        "soapbottle",
        "tissuebox",
        "toiletpaper",
    },
}

OBJECT_CATEGORY_ALIASES = {
    "beverages": "beverage",
    "clean": "cleaning",
    "cleaner": "cleaning",
    "cleaningitems": "cleaning",
    "containers": "container",
    "cooking": "cookware",
    "dishes": "dishware",
    "dish": "dishware",
    "drink": "beverage",
    "drinks": "beverage",
    "drinkware": "dishware",
    "decoration": "decorative",
    "decorations": "decorative",
    "decorativeitem": "decorative",
    "decorativeitems": "decorative",
    "electronics": "electronic",
    "paperwork": "document",
    "personalitem": "personal",
    "personalitems": "personal",
}

MEMBER_TO_OBJECT_CATEGORIES: dict[str, set[str]] = {}
for _category, _members in OBJECT_SEMANTIC_CATEGORIES.items():
    for _member in _members:
        MEMBER_TO_OBJECT_CATEGORIES.setdefault(_member, set()).add(_category)
for _alias, _category in OBJECT_CATEGORY_ALIASES.items():
    if _category in OBJECT_SEMANTIC_CATEGORIES:
        OBJECT_SEMANTIC_CATEGORIES.setdefault(
            _alias, OBJECT_SEMANTIC_CATEGORIES[_category]
        )


def _semantic_trace_line(action: str) -> str:
    line = TARGET_ANNOTATION_RE.sub("", str(action or "")).strip()
    return re.sub(r"\s+", " ", line)


def _append_trace(trace: list[str], action: str) -> None:
    line = _semantic_trace_line(action)
    if line and line not in trace:
        trace.append(line)


def _query_intent(query: str) -> str:
    query_lower = str(query or "").lower()
    query_terms = _terms(query_lower)
    habit_hit = bool(query_terms & HABIT_QUERY_CUES)
    location_hit = bool(query_terms & LOCATION_QUERY_CUES) or any(
        cue in query_lower for cue in LOCATION_QUERY_SUBSTRINGS
    )
    constraint_hit = bool(query_terms & CONSTRAINT_QUERY_CUES) or any(
        cue in query_lower for cue in CONSTRAINT_QUERY_SUBSTRINGS
    )

    explicit_habit_phrase = (
        "household habit" in query_lower
        or "implicit habit" in query_lower
        or "household preference" in query_lower
    )
    if explicit_habit_phrase or ("where should" in query_lower and not constraint_hit):
        return "habit"
    if "where is" in query_lower or "where was" in query_lower:
        return "location"
    if constraint_hit and not habit_hit and not location_hit:
        return "constraint"
    if habit_hit and not location_hit:
        return "habit"
    if constraint_hit and not habit_hit:
        return "constraint"
    if location_hit and not habit_hit:
        return "location"
    if habit_hit and location_hit:
        return "mixed"
    return "generic"


def _source_weight(source: str | None) -> float:
    return SOURCE_WEIGHTS.get(str(source or ""), 0.75)


def _habit_source_authority(source: str | None) -> int:
    return HABIT_SOURCE_AUTHORITY.get(str(source or ""), 0)


def _ranked_counts(counts: dict, *, limit: int = 5) -> list[dict]:
    ranked: list[dict] = []
    for key, value in (counts or {}).items():
        if not key:
            continue
        ranked.append({"key": key, "count": int(value or 0)})
    ranked.sort(key=lambda item: (-item["count"], item["key"]))
    return ranked[:limit]


def _ranked_location_support(
    support: dict,
    *,
    limit: int = 5,
    preferred_key: str | None = None,
) -> list[dict]:
    ranked: list[dict] = []
    for key, stats in (support or {}).items():
        if not key or not isinstance(stats, dict):
            continue
        ranked.append(
            {
                "key": key,
                "count": int(stats.get("count") or 0),
                "last_step": stats.get("last_step"),
                "max_source_weight": float(stats.get("max_source_weight") or 0.0),
                "explicit_count": int(stats.get("explicit_count") or 0),
                "authoritative_continuity_count": int(
                    stats.get("authoritative_continuity_count") or 0
                ),
                "belief_status": stats.get("belief_status") or "unknown",
                "is_current": bool(stats.get("is_current")),
                "superseded_by_target_key": stats.get("superseded_by_target_key"),
                "variants": sorted(stats.get("variants") or []),
            }
        )
    ranked.sort(
        key=lambda item: (
            item["key"] != preferred_key,
            -int(item.get("explicit_count") or 0),
            -int(item.get("authoritative_continuity_count") or 0),
            -_safe_int(item.get("last_step")),
            -item["count"],
            -item["max_source_weight"],
            item["key"],
        )
    )
    return ranked[:limit]


def _support_example_key(example: dict) -> tuple:
    return (
        example.get("object_key"),
        example.get("target_key"),
        example.get("failed_target_key"),
        example.get("step"),
        tuple(example.get("source_event_ids") or []),
    )


def _safe_int(value, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _location_support_is_authoritative(stats: dict | None) -> bool:
    if not isinstance(stats, dict):
        return False
    return bool(
        int(stats.get("explicit_count") or 0) > 0
        or int(stats.get("authoritative_continuity_count") or 0) > 0
    )


def _location_support_sort_key(stats: dict, *, authoritative_mode: bool) -> tuple:
    if authoritative_mode:
        return (
            _safe_int(stats.get("last_step")),
            int(stats.get("authoritative_continuity_count") or 0),
            int(stats.get("explicit_count") or 0),
            float(stats.get("last_source_weight") or 0.0),
            int(stats.get("count") or 0),
            str(stats.get("target_key") or ""),
        )
    return (
        _safe_int(stats.get("last_step")),
        float(stats.get("last_source_weight") or 0.0),
        int(stats.get("count") or 0),
        float(stats.get("max_source_weight") or 0.0),
        str(stats.get("target_key") or ""),
    )


def _select_location_support(target_support: dict) -> dict | None:
    """Select the current location belief from merged target evidence.

    A location cue from the episode description is treated as authoritative,
    but it can still be superseded by an observed movement chain that starts
    from an authoritative target.  This keeps unrelated later noise from
    overwriting memory while allowing real online state changes to update it.
    """
    candidates = [
        stats
        for stats in (target_support or {}).values()
        if isinstance(stats, dict) and stats.get("target_key")
    ]
    if not candidates:
        return None
    authoritative_mode = any(
        _location_support_is_authoritative(stats) for stats in candidates
    )
    if authoritative_mode:
        candidates = [
            stats for stats in candidates if _location_support_is_authoritative(stats)
        ]
    return max(
        candidates,
        key=lambda stats: _location_support_sort_key(
            stats,
            authoritative_mode=authoritative_mode,
        ),
    )


def _select_habit_target_support(target_support: dict) -> dict | None:
    candidates = [
        stats
        for stats in (target_support or {}).values()
        if isinstance(stats, dict) and stats.get("target_key")
    ]
    if not candidates:
        return None
    max_authority = max(
        int(stats.get("max_source_authority") or 0) for stats in candidates
    )
    if max_authority > 0:
        candidates = [
            stats
            for stats in candidates
            if int(stats.get("max_source_authority") or 0) == max_authority
        ]
    return max(
        candidates,
        key=lambda stats: (
            _safe_int(stats.get("last_step")),
            int(stats.get("last_source_authority") or 0),
            int(stats.get("count") or 0),
            float(stats.get("last_source_weight") or 0.0),
            float(stats.get("max_source_weight") or 0.0),
            str(stats.get("target_key") or ""),
        ),
    )


def _is_generic_location_source(target: str | None) -> bool:
    key = _canonical_label(target or "")
    return key in {"", "agentinventory", "floor"}


def _count_terms(target: dict, terms: set[str]) -> None:
    for term in terms:
        if term:
            target[term] = int(target.get(term) or 0) + 1


def _location_object_match(
    item: dict,
    *,
    query_terms: set[str],
    semantic_object_query_terms: set[str],
) -> dict:
    """Classify object-name matches without treating compound pieces as exact.

    Location retrieval needs object identity more than broad semantic recall:
    a query for "Box" should prefer an actual Box over TissueBox, while still
    allowing TissueBox as a fallback when no exact Box memory exists.
    """
    object_key = str(item.get("object_key") or "")
    object_family_key = str(item.get("object_family_key") or "")
    strong_terms = {
        term
        for term in {
            object_key,
            object_family_key,
            *_raw_label_terms(item.get("object") or ""),
            *_raw_label_terms(item.get("object_family") or ""),
        }
        if term
    }
    component_terms = (
        _terms(item.get("object") or "")
        | set((item.get("object_terms") or {}).keys())
        | set((item.get("object_family_terms") or {}).keys())
    ) - strong_terms
    category_terms = _object_categories(
        item.get("object_family") or item.get("object") or ""
    )

    instance_terms = {object_key} if object_key else set()
    family_terms = {object_family_key} if object_family_key else set()
    instance_matches = query_terms & instance_terms
    family_matches = query_terms & family_terms
    strong_matches = query_terms & strong_terms
    component_matches = query_terms & component_terms
    semantic_matches = semantic_object_query_terms & category_terms

    if instance_matches:
        level = 4
    elif family_matches or strong_matches:
        level = 3
    elif component_matches:
        level = 2
    elif semantic_matches:
        level = 1
    else:
        level = 0

    return {
        "level": level,
        "instance_matches": instance_matches,
        "family_matches": family_matches,
        "strong_matches": strong_matches,
        "component_matches": component_matches,
        "semantic_matches": semantic_matches,
        "matched_terms": (
            instance_matches
            | family_matches
            | strong_matches
            | component_matches
            | semantic_matches
        ),
        "strong_terms": strong_terms,
        "component_terms": component_terms,
        "category_terms": category_terms,
    }


def _habit_confidence(group: dict) -> float:
    support_count = int(group.get("support_count") or 0)
    object_count = len(group.get("support_objects") or [])
    negative_count = len(group.get("negative_targets") or [])
    source_weights = group.get("source_weights") or [0.75]
    source_mean = sum(source_weights) / max(len(source_weights), 1)
    source_authorities = group.get("source_authorities") or [0]
    authority_mean = sum(source_authorities) / max(len(source_authorities), 1)
    confidence = (
        0.35
        + 0.15 * min(support_count, 3)
        + 0.08 * min(object_count, 3)
        + 0.04 * min(negative_count, 3)
        + 0.20 * source_mean
        + 0.04 * min(authority_mean, 3) / 3
    )
    return round(min(confidence, 1.0), 4)


def _location_confidence(group: dict) -> float:
    observation_count = int(group.get("observation_count") or 0)
    target_key = group.get("target_key")
    target_family_key = group.get("target_family_key") or target_key
    target_support = (group.get("target_support") or {}).get(target_key, {})
    target_family_support = (group.get("target_family_support") or {}).get(
        target_family_key,
        {},
    )
    same_target_count = int(target_support.get("count") or 0)
    same_family_count = int(target_family_support.get("count") or same_target_count)
    source_weight = float(group.get("last_source_weight") or 0.75)
    confidence = (
        0.30
        + 0.10 * min(observation_count, 3)
        + 0.15 * min(same_family_count, 3)
        + 0.30 * source_weight
    )
    return round(min(confidence, 1.0), 4)


def _failure_confidence(group: dict) -> float:
    support_count = int(group.get("support_count") or 0)
    source_weights = group.get("source_weights") or [0.75]
    source_mean = sum(source_weights) / max(len(source_weights), 1)
    confidence = 0.35 + 0.18 * min(support_count, 3) + 0.25 * source_mean
    return round(min(confidence, 1.0), 4)


def _affordance_confidence(group: dict) -> float:
    support_count = int(group.get("support_count") or 0)
    source_weights = group.get("source_weights") or [0.75]
    source_mean = sum(source_weights) / max(len(source_weights), 1)
    confidence = 0.30 + 0.16 * min(support_count, 3) + 0.26 * source_mean
    return round(min(confidence, 1.0), 4)


def _interaction_state_confidence(group: dict) -> float:
    support_count = int(group.get("success_count") or 0) + int(
        group.get("failure_count") or 0
    )
    conflict_count = int(group.get("conflict_count") or 0)
    confidence = 0.30 + 0.13 * min(support_count, 4)
    if group.get("current_status") in {"available", "blocked"}:
        confidence += 0.18
    confidence -= 0.08 * min(conflict_count, 3)
    return round(max(0.05, min(confidence, 1.0)), 4)


def _interaction_policy(state: dict) -> tuple[set[str], set[str], str | None]:
    """Derive executable action policy from merged interaction evidence."""
    action_type = str(state.get("action_type") or "Interaction")
    if state.get("current_status") == "available":
        return set(), {action_type}, None
    if state.get("current_status") != "blocked":
        return set(), set(), None

    failure_modes = state.get("failure_modes") or {}
    blocked: set[str] = set()
    reasons: list[str] = []
    for mode, count in failure_modes.items():
        if int(count or 0) <= 0:
            continue
        next_blocked = BLOCKED_ACTION_TYPES_BY_FAILURE.get((action_type, str(mode)))
        if next_blocked:
            blocked.update(next_blocked)
            reasons.append(str(mode))
    if not blocked:
        blocked.add(action_type)
        if failure_modes:
            reasons.extend(str(mode) for mode in failure_modes)
    reason = ",".join(dict.fromkeys(reasons)) if reasons else None
    return blocked, set(), reason


def _interaction_attempt(action: str, raw_action: str) -> tuple[str, str] | None:
    text = f"{action or ''} {raw_action or ''}"
    annotated = TARGET_ANNOTATION_RE.search(text)
    for action_type, pattern in INTERACTION_ACTION_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        target = annotated.group("target") if annotated else match.group("target")
        if target:
            return action_type, target
    return None


def _target_variants_from_metadata(metadata: dict, target: str) -> set[str]:
    variants = {str(target or "").strip()}
    for key in (
        "raw_action_target",
        "raw_target_id",
        "action_target",
        "target_id",
        "target",
    ):
        value = metadata.get(key) if isinstance(metadata, dict) else None
        if isinstance(value, str) and value.strip():
            variants.add(value.strip())
    return {value for value in variants if value}


def _interaction_identity_target(metadata: dict, target: str) -> str:
    if isinstance(metadata, dict):
        raw_target = metadata.get("raw_action_target")
        if isinstance(raw_target, str) and raw_target.strip():
            return raw_target.strip()
    return str(target or "").strip()


def _annotated_target(action: str, raw_action: str) -> str | None:
    match = TARGET_ANNOTATION_RE.search(f"{action or ''} {raw_action or ''}")
    if not match:
        return None
    return match.group("target")


def _failure_mode(feedback: str, raw_feedback: str) -> str:
    text = f"{feedback or ''} {raw_feedback or ''}".lower()
    if "locked" in text or "锁" in text:
        return "locked"
    if "not reachable" in text or "unreachable" in text:
        return "unreachable"
    if "no valid positions" in text:
        return "invalid_placement"
    if "not visible" in text:
        return "not_visible"
    if "not accepted" in text or "failed" in text or "cannot" in text:
        return "action_failed"
    return "failed"


class ExperienceMemory:
    """Stores and retrieves consolidated lessons inferred from raw events."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.max_experiences = config.get("max_experiences", 1000)
        self.max_location_observations = config.get(
            "max_location_observations",
            self.max_experiences,
        )
        self.pending_window = config.get("pending_window", 8)
        self.allow_cross_namespace_fallback = config.get(
            "allow_cross_namespace_fallback", False
        )
        self.consolidated_portfolio_only = bool(
            config.get("consolidated_portfolio_only", True)
        )
        self.experiences: List[dict] = []
        self.consolidated: dict[tuple[str, str], dict] = {}
        self.habit_pairs: dict[tuple[str, str, str], dict] = {}
        self.habit_object_beliefs: dict[tuple[str, str], dict] = {}
        self.interaction_failures: List[dict] = []
        self.failure_constraints: dict[tuple[str, str, str], dict] = {}
        self.interaction_affordance_observations: List[dict] = []
        self.interaction_affordances: dict[tuple[str, str, str], dict] = {}
        self.interaction_states: dict[tuple[str, str, str], dict] = {}
        self.location_observations: List[dict] = []
        self.latest_locations: dict[tuple[str, str, str], dict] = {}
        self._pending_failure: Optional[dict] = None
        self._recent_correction: Optional[dict] = None
        self._held_object: Optional[str] = None
        self._held_object_source: Optional[str] = None
        self._last_navigation_target: Optional[str] = None

    def reset(self):
        self.experiences.clear()
        self.consolidated.clear()
        self.habit_pairs.clear()
        self.habit_object_beliefs.clear()
        self.interaction_failures.clear()
        self.failure_constraints.clear()
        self.interaction_affordance_observations.clear()
        self.interaction_affordances.clear()
        self.interaction_states.clear()
        self.location_observations.clear()
        self.latest_locations.clear()
        self._pending_failure = None
        self._recent_correction = None
        self._held_object = None
        self._held_object_source = None
        self._last_navigation_target = None

    def observe_event(self, event: dict) -> None:
        """Update consolidated memory from one raw event."""
        action = str(event.get("action") or "")
        feedback = str(event.get("feedback") or "")
        success = bool(event.get("success"))
        metadata = event.get("metadata") or {}
        raw_action = str(metadata.get("raw_action_text") or action)
        raw_feedback = str(metadata.get("raw_feedback_text") or feedback)
        namespace = self._namespace(metadata)

        pending = self._pending_failure
        if pending:
            pending_namespace = pending.get("namespace")
            if pending_namespace and namespace and pending_namespace != namespace:
                self._pending_failure = None
                pending = None
            elif (
                int(event.get("step") or 0) - int(pending.get("step") or 0)
                > self.pending_window
            ):
                self._pending_failure = None
                pending = None
        recent_correction = self._recent_correction
        if recent_correction:
            recent_namespace = recent_correction.get("namespace")
            if recent_namespace and namespace and recent_namespace != namespace:
                self._recent_correction = None
                recent_correction = None
            elif (
                int(event.get("step") or 0) - int(recent_correction.get("step") or 0)
                > self.pending_window
            ):
                self._recent_correction = None
                recent_correction = None

        placement = PLACEMENT_RE.search(action) or PLACEMENT_RE.search(raw_action)
        placement_semantic_target = placement.group("target") if placement else None
        placement_instance_target = (
            _annotated_target(action, raw_action) or placement_semantic_target
        )
        pickup = PICKUP_RE.search(action) or PICKUP_RE.search(raw_action)
        navigation = NAVIGATE_RE.search(action) or NAVIGATE_RE.search(raw_action)
        text = f"{action} {feedback} {raw_action} {raw_feedback}".lower()
        is_preference_failure = placement is not None and (
            "preference violation" in text or "household habit" in text
        )

        if pending and pending.get("intervened") and placement:
            obj = placement.group("object") or pending.get("object")
            correct_target = placement_semantic_target
            failed_target = pending.get("failed_target")
            pending_object = pending.get("object")
            same_object = (
                not pending_object
                or _singular(obj) == _singular(pending_object)
                or "correct behavior" in text
            )
            candidate_target = pending.get("candidate_target")
            target_matches_navigation = not candidate_target or _singular(
                candidate_target
            ) == _singular(correct_target)
            non_preference_correction_failure = (
                not success
                and "preference violation" not in text
                and "household habit" not in text
            )
            if (
                same_object
                and target_matches_navigation
                and (
                    success
                    or non_preference_correction_failure
                    or "correct behavior" in text
                )
                and obj
                and correct_target
                and failed_target
                and _singular(correct_target) != _singular(failed_target)
            ):
                self.add_corrected_placement(
                    obj=obj,
                    failed_target=failed_target,
                    correct_target=correct_target,
                    physical_target=placement_instance_target,
                    namespace=pending.get("namespace"),
                    metadata=pending.get("metadata") or {},
                    event_ids=[*pending.get("event_ids", []), event.get("id")],
                    step=event.get("step"),
                    source=(
                        "post_intervention_placement"
                        if success
                        else "post_intervention_attempt"
                    ),
                    semantic_trace=[
                        *pending.get("semantic_trace", []),
                        _semantic_trace_line(action),
                    ],
                )
                if not success:
                    self._recent_correction = {
                        "object": obj,
                        "failed_target": failed_target,
                        "correct_target": correct_target,
                        "step": event.get("step"),
                        "namespace": pending.get("namespace"),
                        "metadata": pending.get("metadata") or {},
                        "event_ids": [*pending.get("event_ids", []), event.get("id")],
                        "semantic_trace": [
                            *pending.get("semantic_trace", []),
                            _semantic_trace_line(action),
                        ],
                    }
                else:
                    self._recent_correction = None
                if success:
                    self.add_observed_location(
                        obj=obj,
                        target=placement_instance_target,
                        relation=_placement_relation(placement.group(0)),
                        namespace=pending.get("namespace"),
                        metadata=pending.get("metadata") or {},
                        event_ids=[event.get("id")],
                        step=event.get("step"),
                        source="post_intervention_placement",
                    )
                if success:
                    self._held_object = None
            self._pending_failure = None
            return

        if placement and is_preference_failure:
            self._pending_failure = {
                "object": placement.group("object"),
                "failed_target": placement_semantic_target,
                "step": event.get("step"),
                "namespace": namespace,
                "metadata": metadata,
                "event_ids": [event.get("id")],
                "intervened": False,
                "semantic_trace": [],
            }
            return

        if (
            not success
            and navigation
            and self._held_object
            and not (pending and pending.get("intervened"))
            and (
                "preference violation" in text
                or "failed" in text
                or "wrong" in text
                or "not accepted" in text
            )
        ):
            self._pending_failure = {
                "object": self._held_object,
                "failed_target": navigation.group("target"),
                "step": event.get("step"),
                "namespace": namespace,
                "metadata": metadata,
                "event_ids": [event.get("id")],
                "intervened": False,
                "failure_mode": "navigation_while_holding",
                "semantic_trace": [],
            }
            return

        if pending and "human intervention" in text and success:
            explicit_target = INTERVENTION_TARGET_RE.search(
                f"{raw_action} {raw_feedback} {action} {feedback}"
            )
            if explicit_target:
                obj = explicit_target.group("object") or pending.get("object")
                correct_target = explicit_target.group("target")
                failed_target = pending.get("failed_target")
                if (
                    obj
                    and correct_target
                    and failed_target
                    and _singular(correct_target) != _singular(failed_target)
                ):
                    self.add_corrected_placement(
                        obj=obj,
                        failed_target=failed_target,
                        correct_target=correct_target,
                        physical_target=_annotated_target(action, raw_action),
                        namespace=pending.get("namespace"),
                        metadata=pending.get("metadata") or {},
                        event_ids=[*pending.get("event_ids", []), event.get("id")],
                        step=event.get("step"),
                        source="explicit_human_intervention",
                        semantic_trace=[
                            f"Human corrected {obj} placement to {correct_target}",
                        ],
                    )
                self._pending_failure = None
                return
            pending["intervened"] = True
            pending.setdefault("event_ids", []).append(event.get("id"))
            return

        if pending and pending.get("intervened") and success:
            if navigation:
                pending["candidate_target"] = navigation.group("target")
                pending.setdefault("event_ids", []).append(event.get("id"))
                _append_trace(pending.setdefault("semantic_trace", []), action)
                return

        if pending and pending.get("intervened") and not success and navigation:
            obj = pending.get("object")
            correct_target = navigation.group("target")
            failed_target = pending.get("failed_target")
            if (
                obj
                and correct_target
                and failed_target
                and _singular(correct_target) != _singular(failed_target)
            ):
                self.add_corrected_placement(
                    obj=obj,
                    failed_target=failed_target,
                    correct_target=correct_target,
                    physical_target=_annotated_target(action, raw_action),
                    namespace=pending.get("namespace"),
                    metadata=pending.get("metadata") or {},
                    event_ids=[*pending.get("event_ids", []), event.get("id")],
                    step=event.get("step"),
                    source="post_intervention_navigation_attempt",
                    semantic_trace=[
                        *pending.get("semantic_trace", []),
                        _semantic_trace_line(action),
                    ],
                )
            self._pending_failure = None
            return

        if not success and not is_preference_failure:
            interaction = _interaction_attempt(action, raw_action)
            if interaction:
                action_type, target = interaction
                failure_mode = _failure_mode(feedback, raw_feedback)
                if failure_mode not in DURABLE_FAILURE_MODES:
                    return
                self.add_interaction_failure(
                    action_type=action_type,
                    target=target,
                    failure_mode=failure_mode,
                    namespace=namespace,
                    metadata=metadata,
                    event_ids=[event.get("id")],
                    step=event.get("step"),
                    source="interaction_failure_event",
                )

        if success:
            interaction = _interaction_attempt(action, raw_action)
            if interaction:
                action_type, target = interaction
                if action_type in DURABLE_SUCCESS_ACTIONS:
                    self.add_interaction_affordance(
                        action_type=action_type,
                        target=target,
                        namespace=namespace,
                        metadata=metadata,
                        event_ids=[event.get("id")],
                        step=event.get("step"),
                        source="interaction_success_event",
                    )

        observed_location = OBSERVED_LOCATION_RE.search(
            f"{action} {raw_action} {feedback} {raw_feedback}"
        )
        if success and observed_location and not placement:
            location_source = (
                "explicit_memory_cue"
                if metadata.get("memory_phase")
                in {
                    "context_memory_cue",
                    "context_description_cue",
                }
                else "observed_location_event"
            )
            self.add_observed_location(
                obj=observed_location.group("object"),
                target=observed_location.group("target"),
                relation=_placement_relation(observed_location.group("relation")),
                namespace=namespace,
                metadata=metadata,
                event_ids=[event.get("id")],
                step=event.get("step"),
                source=location_source,
            )

        if navigation and success:
            self._last_navigation_target = _annotated_target(
                action, raw_action
            ) or navigation.group("target")

        if pickup and success:
            picked_object = _annotated_target(action, raw_action) or pickup.group(
                "object"
            )
            self._held_object = picked_object
            self._held_object_source = self._last_navigation_target
            self.add_observed_location(
                obj=picked_object,
                target="AgentInventory",
                relation="held_by_agent",
                namespace=namespace,
                metadata=metadata,
                event_ids=[event.get("id")],
                step=event.get("step"),
                source="successful_pickup",
            )
        elif placement and success:
            placed_object = placement.group("object")
            location_object = placed_object
            previous_target = self._held_object_source
            if self._held_object and _canonical_label(
                self._held_object
            ) == _canonical_label(placed_object):
                location_object = self._held_object
            recent = self._recent_correction
            if recent:
                recent_object = recent.get("object")
                recent_target = recent.get("correct_target")
                same_object = (
                    not recent_object
                    or _canonical_label(recent_object)
                    == _canonical_label(placed_object)
                    or (
                        self._held_object
                        and _canonical_label(recent_object)
                        == _canonical_label(self._held_object)
                    )
                )
                same_target = (
                    recent_target
                    and placement_semantic_target
                    and _canonical_label(recent_target)
                    == _canonical_label(placement_semantic_target)
                )
                if same_object and same_target:
                    self.add_corrected_placement(
                        obj=recent_object or placed_object,
                        failed_target=recent.get("failed_target"),
                        correct_target=recent_target,
                        physical_target=placement_instance_target,
                        namespace=recent.get("namespace"),
                        metadata=recent.get("metadata") or {},
                        event_ids=[*recent.get("event_ids", []), event.get("id")],
                        step=event.get("step"),
                        source="post_intervention_followup_placement",
                        semantic_trace=[
                            *recent.get("semantic_trace", []),
                            _semantic_trace_line(action),
                        ],
                    )
                    self._recent_correction = None
            self.add_observed_location(
                obj=location_object,
                target=placement_instance_target,
                relation=_placement_relation(placement.group(0)),
                namespace=namespace,
                metadata=metadata,
                event_ids=[event.get("id")],
                step=event.get("step"),
                source=(
                    "successful_relocation"
                    if previous_target
                    and not _is_generic_location_source(previous_target)
                    and _canonical_label(previous_target)
                    != _canonical_label(location_object)
                    and _canonical_label(previous_target)
                    != _canonical_label(placement_instance_target)
                    else "successful_placement"
                ),
                previous_target=previous_target,
            )
            if not self._held_object or _singular(placed_object) == _singular(
                self._held_object
            ):
                self._held_object = None
                self._held_object_source = None

    def add_corrected_placement(
        self,
        *,
        obj: str,
        failed_target: str,
        correct_target: str,
        physical_target: str | None = None,
        namespace: str | None,
        metadata: dict,
        event_ids: list[str],
        step: int,
        source: str = "post_intervention_placement",
        semantic_trace: list[str] | None = None,
    ) -> str:
        eid = f"experience_{len(self.experiences)}"
        if source == "post_intervention_navigation_attempt":
            evidence = (
                f"{obj} failed on/in {failed_target}; after human intervention "
                f"the agent was redirected toward {correct_target}."
            )
        else:
            evidence = (
                f"{obj} failed on/in {failed_target}; after human intervention "
                f"it was placed on/in {correct_target}."
            )
        record = {
            "id": eid,
            "type": "corrected_placement",
            "source": source,
            "step": step,
            "object": obj,
            "failed_target": failed_target,
            "correct_target": correct_target,
            "physical_target": physical_target,
            "namespace": namespace,
            "metadata": metadata or {},
            "source_event_ids": event_ids,
            "semantic_trace": [
                line
                for line in (
                    _semantic_trace_line(item) for item in (semantic_trace or [])
                )
                if line
            ],
            "evidence": evidence,
        }
        self.experiences.append(record)
        self._merge_record(record)
        if len(self.experiences) > self.max_experiences:
            self.rebuild_consolidated(self.experiences[1:])
        return eid

    def add_observed_location(
        self,
        *,
        obj: str,
        target: str,
        relation: str,
        namespace: str | None,
        metadata: dict,
        event_ids: list[str],
        step: int,
        source: str = "successful_placement",
        previous_target: str | None = None,
    ) -> str:
        """Store an episodic object-location fact inferred from raw events."""
        oid = f"location_{len(self.location_observations)}"
        obj_label = _instance_label(obj)
        target_label = _instance_label(target)
        previous_target_label = _instance_label(previous_target or "")
        previous_target_family = _base_label(previous_target or "")
        obj_family = _base_label(obj)
        target_family = _base_label(target)
        record = {
            "id": oid,
            "type": "observed_object_location",
            "source": source,
            "step": step,
            "object": obj_label,
            "object_family": obj_family,
            "target": target_label,
            "target_family": target_family,
            "previous_target": previous_target_label or None,
            "previous_target_family": previous_target_family or None,
            "relation": relation or "on_or_in",
            "namespace": namespace,
            "metadata": metadata or {},
            "source_event_ids": event_ids,
            "evidence": (
                f"{obj_label} moved from {previous_target_label} to {target_label}."
                if previous_target_label
                and _canonical_label(previous_target_label)
                != _canonical_label(target_label)
                else (
                    f"{obj_label} was last observed {relation or 'on_or_in'} "
                    f"{target_label}."
                )
            ),
        }
        self.location_observations.append(record)
        self._merge_location_record(record)
        if len(self.location_observations) > self.max_location_observations:
            self.rebuild_location_memory(self.location_observations[1:])
        return oid

    def add_interaction_failure(
        self,
        *,
        action_type: str,
        target: str,
        failure_mode: str,
        namespace: str | None,
        metadata: dict,
        event_ids: list[str],
        step: int,
        source: str = "interaction_failure_event",
    ) -> str:
        """Store a reusable negative interaction fact, e.g. a locked container."""
        fid = f"failure_{len(self.interaction_failures)}"
        target_label = _instance_label(target)
        target_family = _base_label(target)
        record = {
            "id": fid,
            "type": "interaction_failure_constraint",
            "source": source,
            "step": step,
            "action_type": action_type,
            "target": target_label,
            "target_family": target_family,
            "failure_mode": failure_mode or "failed",
            "namespace": namespace,
            "metadata": metadata or {},
            "target_variants": _target_variants_from_metadata(metadata or {}, target),
            "source_event_ids": event_ids,
            "evidence": (
                f"{action_type} failed on {target_label}; "
                f"failure_mode={failure_mode or 'failed'}."
            ),
        }
        self.interaction_failures.append(record)
        self._merge_failure_record(record)
        if len(self.interaction_failures) > self.max_experiences:
            self.rebuild_failure_constraints(self.interaction_failures[1:])
        return fid

    def add_interaction_affordance(
        self,
        *,
        action_type: str,
        target: str,
        namespace: str | None,
        metadata: dict,
        event_ids: list[str],
        step: int,
        source: str = "interaction_success_event",
    ) -> str:
        """Store a reusable positive interaction fact, e.g. an openable container."""
        aid = f"affordance_{len(self.interaction_affordance_observations)}"
        target_label = _instance_label(target)
        target_family = _base_label(target)
        record = {
            "id": aid,
            "type": "interaction_affordance",
            "source": source,
            "step": step,
            "action_type": action_type,
            "target": target_label,
            "target_family": target_family,
            "namespace": namespace,
            "metadata": metadata or {},
            "target_variants": _target_variants_from_metadata(metadata or {}, target),
            "source_event_ids": event_ids,
            "evidence": f"{action_type} succeeded on {target_label}.",
        }
        self.interaction_affordance_observations.append(record)
        self._merge_affordance_record(record)
        if len(self.interaction_affordance_observations) > self.max_experiences:
            self.rebuild_interaction_affordances(
                self.interaction_affordance_observations[1:]
            )
        return aid

    def _merge_record(self, record: dict) -> None:
        namespace = record.get("namespace") or ""
        target = _base_label(record.get("correct_target") or "")
        if not target:
            return
        target_key = _canonical_label(target)
        key = (namespace, target_key)
        group = self.consolidated.setdefault(
            key,
            {
                "id": f"experience_target_{namespace}_{target_key}",
                "_layer": "experience",
                "experience_type": "corrected_placement_habit",
                "namespace": namespace,
                "target": target,
                "target_key": target_key,
                "target_variants": set(),
                "support_count": 0,
                "support_objects": set(),
                "support_object_counts": {},
                "negative_targets": set(),
                "negative_target_counts": {},
                "physical_target_variants": set(),
                "physical_target_terms": {},
                "physical_target_counts": {},
                "preferred_action_types": set(),
                "blocked_target_keys": set(),
                "support_traces": [],
                "support_examples": [],
                "object_terms": {},
                "object_category_counts": {},
                "evidence": [],
                "source_event_ids": [],
                "source_weights": [],
                "source_authorities": [],
                "last_step": None,
                "confidence": 0.0,
            },
        )
        self._merge_habit_group(group, record, target=target, target_key=target_key)

        obj_key = _canonical_label(record.get("object", ""))
        if not obj_key:
            return
        pair_key = (namespace, obj_key, target_key)
        pair_group = self.habit_pairs.setdefault(
            pair_key,
            {
                "id": f"experience_object_target_{namespace}_{obj_key}_{target_key}",
                "_layer": "experience",
                "experience_type": "corrected_placement_habit",
                "habit_scope": "object_target",
                "namespace": namespace,
                "object_focus_key": obj_key,
                "target": target,
                "target_key": target_key,
                "target_variants": set(),
                "support_count": 0,
                "support_objects": set(),
                "support_object_counts": {},
                "negative_targets": set(),
                "negative_target_counts": {},
                "physical_target_variants": set(),
                "physical_target_terms": {},
                "physical_target_counts": {},
                "preferred_action_types": set(),
                "blocked_target_keys": set(),
                "support_traces": [],
                "support_examples": [],
                "object_terms": {},
                "object_category_counts": {},
                "evidence": [],
                "source_event_ids": [],
                "source_weights": [],
                "source_authorities": [],
                "last_step": None,
                "confidence": 0.0,
            },
        )
        self._merge_habit_group(
            pair_group, record, target=target, target_key=target_key
        )
        self._merge_habit_object_belief(
            record,
            object_key=obj_key,
            target=target,
            target_key=target_key,
        )

    def _merge_habit_group(
        self, group: dict, record: dict, *, target: str, target_key: str
    ) -> None:
        obj_key = _canonical_label(record.get("object", ""))
        failed_key = _canonical_label(record.get("failed_target", ""))
        step = _safe_int(record.get("step"))
        group["support_count"] += 1
        group["target_variants"].add(target)
        if obj_key:
            group["support_objects"].add(obj_key)
            group["support_object_counts"][obj_key] = (
                int(group["support_object_counts"].get(obj_key) or 0) + 1
            )
        if failed_key:
            group["negative_targets"].add(failed_key)
            group["negative_target_counts"][failed_key] = (
                int(group["negative_target_counts"].get(failed_key) or 0) + 1
            )
            group["blocked_target_keys"].add(failed_key)
        physical_target = _base_label(record.get("physical_target") or "")
        physical_target_key = _canonical_label(physical_target)
        if physical_target_key and physical_target_key != target_key:
            group["physical_target_variants"].add(physical_target)
            group["physical_target_counts"][physical_target_key] = (
                int(group["physical_target_counts"].get(physical_target_key) or 0) + 1
            )
            for term in {physical_target_key} | _terms(physical_target):
                if term:
                    group["physical_target_terms"][term] = (
                        int(group["physical_target_terms"].get(term) or 0) + 1
                    )
        if target_key:
            group["preferred_action_types"].add("PutObject")
        object_semantic_terms = {obj_key} | _object_semantic_terms(
            record.get("object", "")
        )
        for term in object_semantic_terms:
            if term:
                group["object_terms"][term] = (
                    int(group["object_terms"].get(term) or 0) + 1
                )
                category_key = _semantic_category_key(term)
                if category_key in OBJECT_SEMANTIC_CATEGORIES:
                    group["object_category_counts"][category_key] = (
                        int(group["object_category_counts"].get(category_key) or 0) + 1
                    )
        for line in record.get("semantic_trace") or []:
            semantic_line = _semantic_trace_line(line)
            if semantic_line and semantic_line not in group["support_traces"]:
                group["support_traces"].append(semantic_line)
        support_example = {
            "object": record.get("object"),
            "object_key": obj_key,
            "failed_target": record.get("failed_target"),
            "failed_target_key": failed_key,
            "target": target,
            "target_key": target_key,
            "physical_target": physical_target or None,
            "physical_target_key": physical_target_key or None,
            "semantic_trace": [
                _semantic_trace_line(line)
                for line in (record.get("semantic_trace") or [])
                if _semantic_trace_line(line)
            ][:4],
            "source": record.get("source"),
            "source_weight": _source_weight(record.get("source")),
            "source_authority": _habit_source_authority(record.get("source")),
            "step": record.get("step"),
            "source_event_ids": list(
                dict.fromkeys(record.get("source_event_ids") or [])
            ),
        }
        if support_example.get("object_key") and _support_example_key(
            support_example
        ) not in {_support_example_key(item) for item in group["support_examples"]}:
            group["support_examples"].append(support_example)
            group["support_examples"].sort(
                key=lambda item: (
                    -float(item.get("source_weight") or 0.0),
                    -_safe_int(item.get("step")),
                    str(item.get("object_key") or ""),
                )
            )
            del group["support_examples"][20:]
        if record.get("evidence") and record["evidence"] not in group["evidence"]:
            group["evidence"].append(record["evidence"])
        for event_id in record.get("source_event_ids") or []:
            if event_id not in group["source_event_ids"]:
                group["source_event_ids"].append(event_id)
        group["source_weights"].append(_source_weight(record.get("source")))
        group["source_authorities"].append(
            _habit_source_authority(record.get("source"))
        )
        if step >= _safe_int(group.get("last_step")):
            group["last_step"] = record.get("step")
        group["confidence"] = _habit_confidence(group)

    def _merge_habit_object_belief(
        self,
        record: dict,
        *,
        object_key: str,
        target: str,
        target_key: str,
    ) -> None:
        namespace = record.get("namespace") or ""
        if not object_key or not target_key:
            return
        key = (namespace, object_key)
        belief = self.habit_object_beliefs.setdefault(
            key,
            {
                "id": f"habit_object_belief_{namespace}_{object_key}",
                "_layer": "experience",
                "experience_type": "corrected_placement_object_belief",
                "namespace": namespace,
                "object_key": object_key,
                "current_target": target,
                "current_target_key": target_key,
                "current_confidence": 0.0,
                "target_support": {},
            },
        )
        source_weight = _source_weight(record.get("source"))
        source_authority = _habit_source_authority(record.get("source"))
        step = _safe_int(record.get("step"))
        stats = belief["target_support"].setdefault(
            target_key,
            {
                "target": target,
                "target_key": target_key,
                "count": 0,
                "last_step": None,
                "last_source_weight": 0.0,
                "max_source_weight": 0.0,
                "last_source_authority": 0,
                "max_source_authority": 0,
                "source_event_ids": [],
                "failed_targets": set(),
                "source_weights": [],
                "source_authorities": [],
            },
        )
        stats["count"] += 1
        failed_key = _canonical_label(record.get("failed_target") or "")
        if failed_key:
            stats["failed_targets"].add(failed_key)
        if step >= _safe_int(stats.get("last_step")):
            stats["last_step"] = record.get("step")
            stats["target"] = target
            stats["target_key"] = target_key
            stats["last_source_weight"] = source_weight
            stats["last_source_authority"] = source_authority
        stats["max_source_weight"] = max(
            float(stats.get("max_source_weight") or 0.0),
            source_weight,
        )
        stats["max_source_authority"] = max(
            int(stats.get("max_source_authority") or 0),
            source_authority,
        )
        stats["source_weights"].append(source_weight)
        stats["source_authorities"].append(source_authority)
        for event_id in record.get("source_event_ids") or []:
            if event_id not in stats["source_event_ids"]:
                stats["source_event_ids"].append(event_id)

        selected = _select_habit_target_support(belief.get("target_support") or {})
        if selected:
            selected_key = selected.get("target_key")
            selected_step = _safe_int(selected.get("last_step"))
            for support_key, support_stats in (
                belief.get("target_support") or {}
            ).items():
                if not isinstance(support_stats, dict):
                    continue
                is_current = support_key == selected_key
                support_stats["is_current"] = is_current
                support_stats["belief_status"] = (
                    "active" if is_current else "superseded"
                )
                support_stats["superseded_by_target_key"] = (
                    None if is_current else selected_key
                )
                support_stats["superseded_step"] = (
                    None if is_current else selected.get("last_step")
                )
                if is_current:
                    support_stats["superseded_reason"] = None
                elif int(selected.get("max_source_authority") or 0) > int(
                    support_stats.get("max_source_authority") or 0
                ):
                    support_stats["superseded_reason"] = (
                        "higher_authority_exact_object_belief"
                    )
                elif selected_step > _safe_int(support_stats.get("last_step")):
                    support_stats["superseded_reason"] = "newer_exact_object_correction"
                else:
                    support_stats["superseded_reason"] = (
                        "lower_priority_exact_object_belief"
                    )
            belief["current_target"] = selected.get("target")
            belief["current_target_key"] = selected.get("target_key")
            belief["current_support_count"] = int(selected.get("count") or 0)
            belief["current_last_step"] = selected.get("last_step")
            belief["active_target_support"] = dict(selected)
            belief["current_confidence"] = _habit_confidence(
                {
                    "support_count": selected.get("count"),
                    "support_objects": [object_key],
                    "negative_targets": selected.get("failed_targets") or [],
                    "source_weights": selected.get("source_weights") or [],
                }
            )

    def rebuild_consolidated(self, records: list[dict] | None = None) -> None:
        self.consolidated.clear()
        self.habit_pairs.clear()
        self.habit_object_beliefs.clear()
        if records is not None:
            self.experiences = list(records)
        for record in self.experiences:
            self._merge_record(record)

    def _merge_failure_record(self, record: dict) -> None:
        namespace = record.get("namespace") or ""
        target = _instance_label(record.get("target") or "")
        if not target:
            return
        action_type = str(record.get("action_type") or "Interaction")
        action_key = action_type.lower()
        identity_target = _interaction_identity_target(
            record.get("metadata") or {}, target
        )
        target_key = _identity_key(identity_target)
        target_family = _base_label(record.get("target_family") or target)
        target_family_key = _canonical_label(target_family)
        failure_mode = str(record.get("failure_mode") or "failed")
        target_variants = _target_variants_from_metadata(
            record.get("metadata") or {}, target
        )
        target_variants.update(
            str(value) for value in (record.get("target_variants") or []) if value
        )
        key = (namespace, action_key, target_key)
        group = self.failure_constraints.setdefault(
            key,
            {
                "id": f"experience_failure_{namespace}_{action_key}_{target_key}",
                "_layer": "experience",
                "experience_type": "interaction_failure_constraint",
                "namespace": namespace,
                "action_type": action_type,
                "action_key": action_key,
                "target": target,
                "target_key": target_key,
                "target_family": target_family,
                "target_family_key": target_family_key,
                "target_variants": set(),
                "failure_modes": {},
                "support_count": 0,
                "target_terms": {},
                "evidence": [],
                "source_event_ids": [],
                "source_weights": [],
                "last_step": None,
                "confidence": 0.0,
            },
        )
        group["support_count"] += 1
        group["target_variants"].update(target_variants)
        group["failure_modes"][failure_mode] = (
            int(group["failure_modes"].get(failure_mode) or 0) + 1
        )
        for term in (
            {target_key, target_family_key} | _terms(target) | _terms(target_family)
        ):
            if term:
                group["target_terms"][term] = (
                    int(group["target_terms"].get(term) or 0) + 1
                )
        if record.get("evidence") and record["evidence"] not in group["evidence"]:
            group["evidence"].append(record["evidence"])
        for event_id in record.get("source_event_ids") or []:
            if event_id not in group["source_event_ids"]:
                group["source_event_ids"].append(event_id)
        group["source_weights"].append(_source_weight(record.get("source")))
        step = _safe_int(record.get("step"))
        if step >= _safe_int(group.get("last_step")):
            group["last_step"] = record.get("step")
        group["confidence"] = _failure_confidence(group)
        self._merge_interaction_state(record, outcome="failure")

    def rebuild_failure_constraints(self, records: list[dict] | None = None) -> None:
        self.failure_constraints.clear()
        if records is not None:
            self.interaction_failures = list(records)
        for record in self.interaction_failures:
            self._merge_failure_record(record)
        self.rebuild_interaction_states()

    def _merge_affordance_record(self, record: dict) -> None:
        namespace = record.get("namespace") or ""
        target = _instance_label(record.get("target") or "")
        if not target:
            return
        action_type = str(record.get("action_type") or "Interaction")
        action_key = action_type.lower()
        identity_target = _interaction_identity_target(
            record.get("metadata") or {}, target
        )
        target_key = _identity_key(identity_target)
        target_family = _base_label(record.get("target_family") or target)
        target_family_key = _canonical_label(target_family)
        target_variants = _target_variants_from_metadata(
            record.get("metadata") or {}, target
        )
        target_variants.update(
            str(value) for value in (record.get("target_variants") or []) if value
        )
        key = (namespace, action_key, target_key)
        group = self.interaction_affordances.setdefault(
            key,
            {
                "id": f"experience_affordance_{namespace}_{action_key}_{target_key}",
                "_layer": "experience",
                "experience_type": "interaction_affordance",
                "namespace": namespace,
                "action_type": action_type,
                "action_key": action_key,
                "target": target,
                "target_key": target_key,
                "target_family": target_family,
                "target_family_key": target_family_key,
                "target_variants": set(),
                "support_count": 0,
                "target_terms": {},
                "evidence": [],
                "source_event_ids": [],
                "source_weights": [],
                "last_step": None,
                "confidence": 0.0,
            },
        )
        group["support_count"] += 1
        group["target_variants"].update(target_variants)
        for term in (
            {target_key, target_family_key} | _terms(target) | _terms(target_family)
        ):
            if term:
                group["target_terms"][term] = (
                    int(group["target_terms"].get(term) or 0) + 1
                )
        if record.get("evidence") and record["evidence"] not in group["evidence"]:
            group["evidence"].append(record["evidence"])
        for event_id in record.get("source_event_ids") or []:
            if event_id not in group["source_event_ids"]:
                group["source_event_ids"].append(event_id)
        group["source_weights"].append(_source_weight(record.get("source")))
        step = _safe_int(record.get("step"))
        if step >= _safe_int(group.get("last_step")):
            group["last_step"] = record.get("step")
        group["confidence"] = _affordance_confidence(group)
        self._merge_interaction_state(record, outcome="success")

    def rebuild_interaction_affordances(
        self, records: list[dict] | None = None
    ) -> None:
        self.interaction_affordances.clear()
        if records is not None:
            self.interaction_affordance_observations = list(records)
        for record in self.interaction_affordance_observations:
            self._merge_affordance_record(record)
        self.rebuild_interaction_states()

    def _merge_interaction_state(self, record: dict, *, outcome: str) -> None:
        namespace = record.get("namespace") or ""
        target = _instance_label(record.get("target") or "")
        if not target:
            return
        action_type = str(record.get("action_type") or "Interaction")
        action_key = action_type.lower()
        identity_target = _interaction_identity_target(
            record.get("metadata") or {}, target
        )
        target_key = _identity_key(identity_target)
        target_family = _base_label(record.get("target_family") or target)
        target_family_key = _canonical_label(target_family)
        target_variants = _target_variants_from_metadata(
            record.get("metadata") or {}, target
        )
        target_variants.update(
            str(value) for value in (record.get("target_variants") or []) if value
        )
        key = (namespace, action_key, target_key)
        state = self.interaction_states.setdefault(
            key,
            {
                "id": f"interaction_state_{namespace}_{action_key}_{target_key}",
                "_layer": "experience",
                "experience_type": "interaction_state",
                "namespace": namespace,
                "action_type": action_type,
                "action_key": action_key,
                "target": target,
                "target_key": target_key,
                "target_family": target_family,
                "target_family_key": target_family_key,
                "target_variants": set(),
                "target_terms": {},
                "success_count": 0,
                "failure_count": 0,
                "failure_modes": {},
                "first_step": None,
                "last_step": None,
                "last_outcome": None,
                "current_status": "unknown",
                "blocked_action_types": set(),
                "available_action_types": set(),
                "status_reason": None,
                "conflict_count": 0,
                "evidence": [],
                "source_event_ids": [],
                "confidence": 0.0,
            },
        )
        state["target_variants"].update(target_variants)
        for term in (
            {target_key, target_family_key} | _terms(target) | _terms(target_family)
        ):
            if term:
                state["target_terms"][term] = (
                    int(state["target_terms"].get(term) or 0) + 1
                )
        if outcome == "success":
            state["success_count"] += 1
            next_status = "available"
        else:
            state["failure_count"] += 1
            mode = str(record.get("failure_mode") or "failed")
            state["failure_modes"][mode] = (
                int(state["failure_modes"].get(mode) or 0) + 1
            )
            next_status = "blocked"
        step = _safe_int(record.get("step"))
        if state.get("first_step") is None or step < _safe_int(state.get("first_step")):
            state["first_step"] = record.get("step")
        if state.get("last_outcome") and state.get("last_outcome") != outcome:
            state["conflict_count"] += 1
        if step >= _safe_int(state.get("last_step")):
            state["last_step"] = record.get("step")
            state["last_outcome"] = outcome
            state["current_status"] = next_status
            state["target"] = target
            state["target_family"] = target_family
            state["target_family_key"] = target_family_key
        if record.get("evidence") and record["evidence"] not in state["evidence"]:
            state["evidence"].append(record["evidence"])
        for event_id in record.get("source_event_ids") or []:
            if event_id not in state["source_event_ids"]:
                state["source_event_ids"].append(event_id)
        blocked_actions, available_actions, status_reason = _interaction_policy(state)
        state["blocked_action_types"] = blocked_actions
        state["available_action_types"] = available_actions
        state["status_reason"] = status_reason
        state["confidence"] = _interaction_state_confidence(state)
        self._sync_interaction_state_to_raw_indexes(state)

    def _sync_interaction_state_to_raw_indexes(self, state: dict) -> None:
        """Persist the conflict-resolved policy on raw interaction indexes."""
        key = (
            state.get("namespace") or "",
            state.get("action_key") or "",
            state.get("target_key") or "",
        )
        current_status = state.get("current_status") or "unknown"
        failure = self.failure_constraints.get(key)
        if failure is not None:
            is_current = current_status == "blocked"
            failure["current_status"] = current_status
            failure["is_current_policy"] = is_current
            failure["superseded_by_success"] = current_status == "available"
            failure["blocked_action_types"] = set(
                state.get("blocked_action_types") or []
            )
            failure["available_action_types"] = set(
                state.get("available_action_types") or []
            )
            failure["interaction_state_id"] = state.get("id")
            failure["success_count"] = int(state.get("success_count") or 0)
            failure["failure_count"] = int(state.get("failure_count") or 0)
            failure["conflict_count"] = int(state.get("conflict_count") or 0)
        affordance = self.interaction_affordances.get(key)
        if affordance is not None:
            is_current = current_status == "available"
            affordance["current_status"] = current_status
            affordance["is_current_policy"] = is_current
            affordance["conflicted_by_failure"] = current_status == "blocked"
            affordance["blocked_action_types"] = set(
                state.get("blocked_action_types") or []
            )
            affordance["available_action_types"] = set(
                state.get("available_action_types") or []
            )
            affordance["interaction_state_id"] = state.get("id")
            affordance["success_count"] = int(state.get("success_count") or 0)
            affordance["failure_count"] = int(state.get("failure_count") or 0)
            affordance["conflict_count"] = int(state.get("conflict_count") or 0)

    def rebuild_interaction_states(self) -> None:
        self.interaction_states.clear()
        records: list[tuple[int, str, dict]] = []
        for record in self.interaction_failures:
            records.append((_safe_int(record.get("step")), "failure", record))
        for record in self.interaction_affordance_observations:
            records.append((_safe_int(record.get("step")), "success", record))
        for _, outcome, record in sorted(records, key=lambda item: item[0]):
            self._merge_interaction_state(record, outcome=outcome)

    def _merge_location_record(self, record: dict) -> None:
        namespace = record.get("namespace") or ""
        obj = _instance_label(record.get("object") or "")
        if not obj:
            return
        metadata = record.get("metadata") or {}
        object_key = _instance_key(obj)
        object_family = _base_label(record.get("object_family") or obj)
        object_family_key = _canonical_label(object_family)
        target = _instance_label(record.get("target") or "")
        target_key = _instance_key(target)
        target_family = _base_label(record.get("target_family") or target)
        target_family_key = _canonical_label(target_family)
        room_type = str(metadata.get("room_type") or metadata.get("room") or "")
        scene = str(metadata.get("scene") or "")
        session_name = str(metadata.get("session_name") or "")
        source_context_terms = _terms(
            " ".join(value for value in (room_type, scene, session_name) if value)
        )
        key = (namespace, scene, object_key)
        group = self.latest_locations.setdefault(
            key,
            {
                "id": f"experience_location_{namespace}_{scene}_{object_key}",
                "_layer": "experience",
                "experience_type": "observed_object_location",
                "namespace": namespace,
                "object": obj,
                "object_key": object_key,
                "object_family": object_family,
                "object_family_key": object_family_key,
                "target": target,
                "target_key": target_key,
                "target_family": target_family,
                "target_family_key": target_family_key,
                "relation": record.get("relation") or "on_or_in",
                "last_step": record.get("step"),
                "last_source_weight": _source_weight(record.get("source")),
                "observation_count": 0,
                "object_variants": set(),
                "target_variants": set(),
                "object_terms": {},
                "object_family_terms": {},
                "target_family_terms": {},
                "target_support": {},
                "target_family_support": {},
                "source_context_terms": {},
                "source_rooms": {},
                "source_scenes": {},
                "source_sessions": {},
                "last_room_type": None,
                "last_scene": None,
                "last_session_name": None,
                "movement_count": 0,
                "last_previous_target": None,
                "last_previous_target_key": None,
                "movement_history": [],
                "location_history": [],
                "evidence": [],
                "source_event_ids": [],
                "confidence": 0.0,
            },
        )
        group["observation_count"] += 1
        group["object_variants"].add(obj)
        for term in {object_key, object_family_key} | _terms(obj):
            if term:
                group["object_terms"][term] = (
                    int(group["object_terms"].get(term) or 0) + 1
                )
        for term in {object_family_key} | _terms(object_family):
            if term:
                group["object_family_terms"][term] = (
                    int(group["object_family_terms"].get(term) or 0) + 1
                )
        if target:
            group["target_variants"].add(target)
        for term in {target_family_key} | _terms(target_family):
            if term:
                group["target_family_terms"][term] = (
                    int(group["target_family_terms"].get(term) or 0) + 1
                )
        _count_terms(group["source_context_terms"], source_context_terms)
        if room_type:
            group["source_rooms"][room_type] = (
                int(group["source_rooms"].get(room_type) or 0) + 1
            )
        if scene:
            group["source_scenes"][scene] = (
                int(group["source_scenes"].get(scene) or 0) + 1
            )
        if session_name:
            group["source_sessions"][session_name] = (
                int(group["source_sessions"].get(session_name) or 0) + 1
            )
        source_weight = _source_weight(record.get("source"))
        previous_target = _instance_label(record.get("previous_target") or "")
        previous_target_key = _instance_key(previous_target)
        moved_between_targets = bool(
            previous_target_key
            and not _is_generic_location_source(previous_target)
            and _canonical_label(previous_target) != object_family_key
            and target_key
            and _canonical_label(previous_target) != _canonical_label(target)
        )
        previous_target_stats = (group.get("target_support") or {}).get(
            previous_target_key
        )
        authoritative_continuity = bool(
            moved_between_targets
            and _location_support_is_authoritative(previous_target_stats)
        )
        explicit_observation = record.get("source") == "explicit_memory_cue"
        target_stats = group["target_support"].setdefault(
            target_key,
            {
                "target": target,
                "target_key": target_key,
                "target_family": target_family,
                "target_family_key": target_family_key,
                "count": 0,
                "last_step": None,
                "last_source_weight": 0.0,
                "max_source_weight": 0.0,
                "explicit_count": 0,
                "authoritative_continuity_count": 0,
                "variants": set(),
                "relation": record.get("relation") or "on_or_in",
                "previous_target": None,
                "previous_target_key": None,
                "room_type": None,
                "scene": None,
                "session_name": None,
            },
        )
        target_stats["count"] += 1
        if explicit_observation:
            target_stats["explicit_count"] += 1
        if authoritative_continuity:
            target_stats["authoritative_continuity_count"] += 1
        if target:
            target_stats["variants"].add(target)
        next_step = _safe_int(record.get("step"))
        if next_step >= _safe_int(target_stats.get("last_step")):
            target_stats["last_step"] = record.get("step")
            target_stats["target"] = target
            target_stats["target_key"] = target_key
            target_stats["target_family"] = target_family
            target_stats["target_family_key"] = target_family_key
            target_stats["relation"] = record.get("relation") or "on_or_in"
            target_stats["previous_target"] = previous_target or None
            target_stats["previous_target_key"] = previous_target_key or None
            target_stats["room_type"] = room_type or None
            target_stats["scene"] = scene or None
            target_stats["session_name"] = session_name or None
            target_stats["last_source_weight"] = source_weight
        target_stats["max_source_weight"] = max(
            float(target_stats.get("max_source_weight") or 0.0),
            source_weight,
        )
        target_family_stats = group["target_family_support"].setdefault(
            target_family_key,
            {
                "count": 0,
                "last_step": None,
                "max_source_weight": 0.0,
                "explicit_count": 0,
                "authoritative_continuity_count": 0,
                "variants": set(),
            },
        )
        target_family_stats["count"] += 1
        if explicit_observation:
            target_family_stats["explicit_count"] += 1
        if authoritative_continuity:
            target_family_stats["authoritative_continuity_count"] += 1
        if target:
            target_family_stats["variants"].add(target)
        if next_step >= _safe_int(target_family_stats.get("last_step")):
            target_family_stats["last_step"] = record.get("step")
        target_family_stats["max_source_weight"] = max(
            float(target_family_stats.get("max_source_weight") or 0.0),
            source_weight,
        )
        if moved_between_targets:
            group["movement_count"] += 1
            movement_item = {
                "step": record.get("step"),
                "previous_target": previous_target,
                "previous_target_key": previous_target_key,
                "target": target,
                "target_key": target_key,
                "source": record.get("source"),
            }
            if movement_item not in group["movement_history"]:
                group["movement_history"].append(movement_item)
        history_item = {
            "step": record.get("step"),
            "target": target,
            "target_key": target_key,
            "target_family": target_family,
            "target_family_key": target_family_key,
            "previous_target": previous_target or None,
            "previous_target_key": previous_target_key or None,
            "relation": record.get("relation") or "on_or_in",
            "source": record.get("source"),
            "room_type": room_type or None,
            "scene": scene or None,
            "session_name": session_name or None,
        }
        if history_item not in group["location_history"]:
            group["location_history"].append(history_item)
        if record.get("evidence") and record["evidence"] not in group["evidence"]:
            group["evidence"].append(record["evidence"])
        for event_id in record.get("source_event_ids") or []:
            if event_id not in group["source_event_ids"]:
                group["source_event_ids"].append(event_id)
        selected_target = _select_location_support(group.get("target_support") or {})
        if selected_target:
            selected_key = selected_target.get("target_key")
            selected_step = _safe_int(selected_target.get("last_step"))
            for support_key, support_stats in (
                group.get("target_support") or {}
            ).items():
                if not isinstance(support_stats, dict):
                    continue
                is_current = support_key == selected_key
                support_stats["is_current"] = is_current
                support_stats["belief_status"] = (
                    "active" if is_current else "superseded"
                )
                support_stats["superseded_by_target_key"] = (
                    None if is_current else selected_key
                )
                support_stats["superseded_step"] = (
                    None if is_current else selected_target.get("last_step")
                )
                if is_current:
                    support_stats["superseded_reason"] = None
                elif selected_step > _safe_int(support_stats.get("last_step")):
                    support_stats["superseded_reason"] = "newer_authoritative_location"
                else:
                    support_stats["superseded_reason"] = (
                        "lower_priority_location_belief"
                    )
            group["target"] = selected_target.get("target")
            group["target_key"] = selected_target.get("target_key")
            group["target_family"] = selected_target.get("target_family")
            group["target_family_key"] = selected_target.get("target_family_key")
            group["relation"] = selected_target.get("relation") or "on_or_in"
            group["last_step"] = selected_target.get("last_step")
            group["last_source_weight"] = float(
                selected_target.get("last_source_weight") or 0.0
            )
            group["last_previous_target"] = selected_target.get("previous_target")
            group["last_previous_target_key"] = selected_target.get(
                "previous_target_key"
            )
            group["last_room_type"] = selected_target.get("room_type")
            group["last_scene"] = selected_target.get("scene")
            group["last_session_name"] = selected_target.get("session_name")
        group["confidence"] = _location_confidence(group)

    def rebuild_location_memory(self, records: list[dict] | None = None) -> None:
        self.latest_locations.clear()
        if records is not None:
            self.location_observations = list(records)
        for record in self.location_observations:
            self._merge_location_record(record)

    def query(
        self,
        query: str = "",
        top_k: int = 5,
        namespace: str = None,
        memory_types: set[str] | None = None,
    ) -> List[dict]:
        """Retrieve merged experience candidates."""
        query_terms = _terms(query)
        exact_object_query_terms, semantic_object_query_terms = (
            _expanded_object_query_terms(query)
        )
        query_intent = _query_intent(query)
        include_interaction_states = bool(
            memory_types and "interaction_state" in memory_types
        )
        pool = (
            list(self.consolidated.values())
            + list(self.latest_locations.values())
            + list(self.failure_constraints.values())
            + list(self.interaction_affordances.values())
        )
        if include_interaction_states:
            pool += list(self.interaction_states.values())
        pair_pool = list(self.habit_pairs.values())
        if memory_types:
            pool = [
                item for item in pool if item.get("experience_type") in memory_types
            ]
            if "corrected_placement_habit" not in memory_types:
                pair_pool = []
        if namespace:
            scoped = [item for item in pool if item.get("namespace") == namespace]
            scoped_pairs = [
                item for item in pair_pool if item.get("namespace") == namespace
            ]
            if scoped or scoped_pairs:
                pool = scoped
                pair_pool = scoped_pairs
            elif not self.allow_cross_namespace_fallback:
                return []
        if not pool:
            return []

        current_exact_object_targets = {
            belief.get("current_target_key")
            for belief in self.habit_object_beliefs.values()
            if (not namespace or belief.get("namespace") == namespace)
            and belief.get("object_key") in exact_object_query_terms
            and belief.get("current_target_key")
        }
        exact_object_belief_support_by_target: dict[str, dict] = {}
        for belief in self.habit_object_beliefs.values():
            if namespace and belief.get("namespace") != namespace:
                continue
            if belief.get("object_key") not in exact_object_query_terms:
                continue
            for target_key, stats in (belief.get("target_support") or {}).items():
                if target_key and isinstance(stats, dict):
                    exact_object_belief_support_by_target[target_key] = stats
        pair_target_matches: dict[str, dict] = {}
        latest_exact_pair_step = -1
        for pair in pair_pool:
            pair_terms = set(pair.get("support_objects") or []) | set(
                (pair.get("object_terms") or {}).keys()
            )
            exact_matched_terms = exact_object_query_terms & pair_terms
            semantic_matched_terms = semantic_object_query_terms & pair_terms
            if not exact_matched_terms and not semantic_matched_terms:
                continue
            target_key = pair.get("target_key")
            if not target_key:
                continue
            match = pair_target_matches.setdefault(
                target_key,
                {
                    "support_count": 0,
                    "confidence": 0.0,
                    "last_step": -1,
                    "matched_terms": set(),
                    "matched_exact_terms": set(),
                    "matched_semantic_terms": set(),
                    "support_objects": set(),
                    "exact_support_count": 0,
                    "semantic_support_count": 0,
                    "last_exact_step": -1,
                    "last_semantic_step": -1,
                },
            )
            match["support_count"] += int(pair.get("support_count") or 0)
            match["confidence"] = max(
                float(match.get("confidence") or 0.0),
                float(pair.get("confidence") or 0.0),
            )
            match["last_step"] = max(
                int(match.get("last_step") or -1),
                _safe_int(pair.get("last_step")),
            )
            match["matched_terms"].update(exact_matched_terms | semantic_matched_terms)
            match["matched_exact_terms"].update(exact_matched_terms)
            match["matched_semantic_terms"].update(semantic_matched_terms)
            match["support_objects"].update(pair.get("support_objects") or [])
            pair_support_count = int(pair.get("support_count") or 0)
            pair_step = _safe_int(pair.get("last_step"))
            if exact_matched_terms:
                match["exact_support_count"] += pair_support_count
                match["last_exact_step"] = max(
                    int(match.get("last_exact_step") or -1),
                    pair_step,
                )
                latest_exact_pair_step = max(latest_exact_pair_step, pair_step)
            if semantic_matched_terms:
                match["semantic_support_count"] += pair_support_count
                match["last_semantic_step"] = max(
                    int(match.get("last_semantic_step") or -1),
                    pair_step,
                )
        for target_key, match in pair_target_matches.items():
            exact_step = int(match.get("last_exact_step") or -1)
            exact_recency_adjustment = 0.0
            superseded_by_recent_exact = False
            if exact_step >= 0 and current_exact_object_targets:
                if target_key in current_exact_object_targets:
                    exact_recency_adjustment = 8.0
                else:
                    exact_recency_adjustment = -4.0
                    superseded_by_recent_exact = True
            elif exact_step >= 0 and latest_exact_pair_step >= 0:
                if exact_step == latest_exact_pair_step:
                    exact_recency_adjustment = 8.0
                else:
                    exact_recency_adjustment = -4.0
                    superseded_by_recent_exact = True
            belief_support = exact_object_belief_support_by_target.get(target_key)
            if belief_support:
                match["object_belief_status"] = (
                    belief_support.get("belief_status") or "unknown"
                )
                match["object_belief_superseded_by_target_key"] = belief_support.get(
                    "superseded_by_target_key"
                )
                match["object_belief_max_source_authority"] = int(
                    belief_support.get("max_source_authority") or 0
                )
                match["object_belief_last_source_authority"] = int(
                    belief_support.get("last_source_authority") or 0
                )
                if belief_support.get("belief_status") == "superseded":
                    exact_recency_adjustment = min(exact_recency_adjustment, -4.0)
                    superseded_by_recent_exact = True
            else:
                match["object_belief_status"] = None
                match["object_belief_superseded_by_target_key"] = None
                match["object_belief_max_source_authority"] = None
                match["object_belief_last_source_authority"] = None
            match["superseded_by_recent_exact_match"] = superseded_by_recent_exact
            match["current_exact_object_target"] = (
                target_key in current_exact_object_targets
            )
            match["exact_recency_adjustment"] = exact_recency_adjustment
            match["specificity_score"] = (
                4.0 * len(match.get("matched_exact_terms") or [])
                + 1.25 * len(match.get("matched_semantic_terms") or [])
                + min(int(match.get("support_count") or 0), 5)
                + float(match.get("confidence") or 0.0)
                + 0.001 * max(int(match.get("last_step") or 0), 0)
                + exact_recency_adjustment
            )
        target_competition = sorted(
            (
                {
                    "target_key": target_key,
                    "support_count": int(match.get("support_count") or 0),
                    "confidence": float(match.get("confidence") or 0.0),
                    "matched_terms": sorted(match.get("matched_terms") or []),
                    "matched_exact_terms": sorted(
                        match.get("matched_exact_terms") or []
                    ),
                    "matched_semantic_terms": sorted(
                        match.get("matched_semantic_terms") or []
                    ),
                    "support_objects": sorted(match.get("support_objects") or []),
                    "exact_support_count": int(match.get("exact_support_count") or 0),
                    "semantic_support_count": int(
                        match.get("semantic_support_count") or 0
                    ),
                    "last_exact_step": int(match.get("last_exact_step") or -1),
                    "last_semantic_step": int(match.get("last_semantic_step") or -1),
                    "current_exact_object_target": bool(
                        match.get("current_exact_object_target")
                    ),
                    "object_belief_status": match.get("object_belief_status"),
                    "object_belief_superseded_by_target_key": match.get(
                        "object_belief_superseded_by_target_key"
                    ),
                    "object_belief_max_source_authority": match.get(
                        "object_belief_max_source_authority"
                    ),
                    "object_belief_last_source_authority": match.get(
                        "object_belief_last_source_authority"
                    ),
                    "superseded_by_recent_exact_match": bool(
                        match.get("superseded_by_recent_exact_match")
                    ),
                    "specificity_score": float(match.get("specificity_score") or 0.0),
                }
                for target_key, match in pair_target_matches.items()
            ),
            key=lambda item: (-item["specificity_score"], item["target_key"]),
        )
        target_rank_by_key = {
            item["target_key"]: rank
            for rank, item in enumerate(target_competition, start=1)
        }
        has_exact_pair_match = any(
            int(match.get("exact_support_count") or 0) > 0
            for match in pair_target_matches.values()
        )
        best_specificity = (
            float(target_competition[0]["specificity_score"])
            if target_competition
            else 0.0
        )
        second_specificity = (
            float(target_competition[1]["specificity_score"])
            if len(target_competition) > 1
            else 0.0
        )
        habit_object_overlap_available = any(
            item.get("experience_type") == "corrected_placement_habit"
            and bool(
                exact_object_query_terms
                & (
                    set(item.get("support_objects") or [])
                    | set((item.get("object_terms") or {}).keys())
                )
            )
            for item in pool
        )
        location_object_matches = [
            _location_object_match(
                item,
                query_terms=query_terms,
                semantic_object_query_terms=semantic_object_query_terms,
            )
            for item in pool
            if item.get("experience_type") == "observed_object_location"
        ]
        location_exact_object_match_available = any(
            int(match.get("level") or 0) >= 3 for match in location_object_matches
        )
        location_object_overlap_available = any(
            int(match.get("level") or 0) > 0 for match in location_object_matches
        )
        location_context_overlap_available = any(
            item.get("experience_type") == "observed_object_location"
            and bool(query_terms & set((item.get("source_context_terms") or {}).keys()))
            for item in pool
        )
        failure_by_target = {
            (
                item.get("namespace") or "",
                item.get("action_key") or "",
                item.get("target_key") or "",
            ): item
            for item in pool
            if item.get("experience_type") == "interaction_failure_constraint"
        }
        affordance_by_target = {
            (
                item.get("namespace") or "",
                item.get("action_key") or "",
                item.get("target_key") or "",
            ): item
            for item in pool
            if item.get("experience_type") == "interaction_affordance"
        }
        interaction_state_by_target = dict(self.interaction_states)
        ranked: list[tuple[float, dict]] = []
        for item in pool:
            experience_type = item.get("experience_type")
            evidence_terms = _terms(" ".join(item.get("evidence", []) or []))
            if experience_type == "observed_object_location":
                object_match = _location_object_match(
                    item,
                    query_terms=query_terms,
                    semantic_object_query_terms=semantic_object_query_terms,
                )
                object_match_level = int(object_match.get("level") or 0)
                target_terms = (
                    {item.get("target_key", "")}
                    | {item.get("target_family_key", "")}
                    | _terms(item.get("target", ""))
                    | set((item.get("target_family_terms") or {}).keys())
                )
                instance_matches = set(object_match.get("instance_matches") or [])
                family_matches = set(object_match.get("family_matches") or [])
                strong_matches = set(object_match.get("strong_matches") or [])
                component_matches = set(object_match.get("component_matches") or [])
                semantic_matches = set(object_match.get("semantic_matches") or [])
                strong_object_overlap = len(
                    instance_matches | family_matches | strong_matches
                )
                component_object_overlap = len(component_matches)
                semantic_object_overlap = len(semantic_matches)
                object_overlap = (
                    strong_object_overlap
                    + component_object_overlap
                    + semantic_object_overlap
                )
                target_overlap = len(query_terms & target_terms)
                evidence_overlap = len(query_terms & evidence_terms)
                context_overlap = len(
                    query_terms & set((item.get("source_context_terms") or {}).keys())
                )
                movement_count = int(item.get("movement_count") or 0)
                score = (
                    165 * len(instance_matches)
                    + 135 * len(family_matches | strong_matches)
                    + 45 * component_object_overlap
                    + 28 * semantic_object_overlap
                    + 8 * target_overlap
                    + 2 * evidence_overlap
                    + 90 * min(context_overlap, 2)
                    + 12 * float(item.get("confidence") or 0)
                    + 3 * min(int(item.get("observation_count") or 0), 5)
                    + 90 * min(movement_count, 2)
                    + 0.01 * int(item.get("last_step") or 0)
                )
                if float(item.get("last_source_weight") or 0.0) >= _source_weight(
                    "explicit_memory_cue"
                ):
                    score += 240
                if not query_terms:
                    score += 10
                if query_intent == "location":
                    score += 80
                    if movement_count:
                        score += 45
                    if location_exact_object_match_available and object_match_level < 3:
                        continue
                    if location_object_overlap_available and not object_overlap:
                        continue
                    if location_context_overlap_available:
                        if context_overlap:
                            score += 120
                        else:
                            score -= 80
                elif query_intent == "habit":
                    score *= 0.20
                elif query_intent == "mixed":
                    score *= 0.75
                if (
                    query_terms
                    and not (object_overlap or target_overlap or evidence_overlap)
                    and query_intent != "location"
                ):
                    continue
            elif experience_type == "interaction_state":
                target_terms = (
                    {item.get("target_key", "")}
                    | {item.get("target_family_key", "")}
                    | _terms(item.get("target", ""))
                    | set((item.get("target_terms") or {}).keys())
                )
                action_terms = {
                    str(item.get("action_type") or "").lower(),
                    str(item.get("action_key") or "").lower(),
                }
                status = str(item.get("current_status") or "unknown")
                mode_terms = set((item.get("failure_modes") or {}).keys())
                policy_terms = {
                    status,
                    "avoid" if status == "blocked" else "usable",
                    "blocked" if status == "blocked" else "available",
                    *[
                        str(value).lower()
                        for value in (item.get("blocked_action_types") or [])
                    ],
                    *[
                        str(value).lower()
                        for value in (item.get("available_action_types") or [])
                    ],
                }
                if str(item.get("action_type") or "") == "Open":
                    policy_terms.update({"container", "openable", "candidate"})
                target_overlap = len(query_terms & target_terms)
                action_overlap = len(query_terms & action_terms)
                mode_overlap = len(query_terms & mode_terms)
                policy_overlap = len(query_terms & policy_terms)
                evidence_overlap = len(query_terms & evidence_terms)
                support_count = int(item.get("success_count") or 0) + int(
                    item.get("failure_count") or 0
                )
                score = (
                    105 * target_overlap
                    + 50 * action_overlap
                    + 50 * mode_overlap
                    + 60 * policy_overlap
                    + 8 * evidence_overlap
                    + 9 * min(support_count, 5)
                    + 34 * float(item.get("confidence") or 0)
                    + 0.01 * int(item.get("last_step") or 0)
                )
                if query_intent == "constraint":
                    score += 155
                    if status == "blocked":
                        score += 35
                    if not (
                        target_overlap
                        or action_overlap
                        or mode_overlap
                        or policy_overlap
                        or evidence_overlap
                    ):
                        score += 30
                elif query_intent == "habit":
                    score *= 0.20
                elif query_intent == "location":
                    score *= 0.45
                if int(item.get("conflict_count") or 0) > 0:
                    score *= 0.96
                if (
                    query_terms
                    and not (
                        target_overlap
                        or action_overlap
                        or mode_overlap
                        or policy_overlap
                        or evidence_overlap
                    )
                    and query_intent != "constraint"
                ):
                    continue
            elif experience_type == "interaction_failure_constraint":
                target_terms = (
                    {item.get("target_key", "")}
                    | {item.get("target_family_key", "")}
                    | _terms(item.get("target", ""))
                    | set((item.get("target_terms") or {}).keys())
                )
                action_terms = {
                    str(item.get("action_type") or "").lower(),
                    str(item.get("action_key") or "").lower(),
                }
                mode_terms = set((item.get("failure_modes") or {}).keys()) | {
                    "failure",
                    "failed",
                    "avoid",
                }
                target_overlap = len(query_terms & target_terms)
                action_overlap = len(query_terms & action_terms)
                mode_overlap = len(query_terms & mode_terms)
                evidence_overlap = len(query_terms & evidence_terms)
                support_count = int(item.get("support_count") or 0)
                score = (
                    95 * target_overlap
                    + 45 * action_overlap
                    + 45 * mode_overlap
                    + 8 * evidence_overlap
                    + 10 * min(support_count, 5)
                    + 30 * float(item.get("confidence") or 0)
                    + 0.01 * int(item.get("last_step") or 0)
                )
                if query_intent == "constraint":
                    score += 140
                    if not (
                        target_overlap
                        or action_overlap
                        or mode_overlap
                        or evidence_overlap
                    ):
                        score += 35
                elif query_intent == "habit":
                    score *= 0.20
                elif query_intent == "location":
                    score *= 0.45
                counterpart = affordance_by_target.get(
                    (
                        item.get("namespace") or "",
                        item.get("action_key") or "",
                        item.get("target_key") or "",
                    )
                )
                state = interaction_state_by_target.get(
                    (
                        item.get("namespace") or "",
                        item.get("action_key") or "",
                        item.get("target_key") or "",
                    )
                )
                if counterpart and _safe_int(counterpart.get("last_step")) > _safe_int(
                    item.get("last_step")
                ):
                    score *= 0.35
                if state:
                    if state.get("current_status") == "blocked":
                        score *= 1.15
                    elif state.get("current_status") == "available":
                        score *= 0.35
                    if int(state.get("conflict_count") or 0) > 0:
                        score *= 0.90
                if (
                    query_terms
                    and not (
                        target_overlap
                        or action_overlap
                        or mode_overlap
                        or evidence_overlap
                    )
                    and query_intent != "constraint"
                ):
                    continue
            elif experience_type == "interaction_affordance":
                target_terms = (
                    {item.get("target_key", "")}
                    | {item.get("target_family_key", "")}
                    | _terms(item.get("target", ""))
                    | set((item.get("target_terms") or {}).keys())
                )
                action_terms = {
                    str(item.get("action_type") or "").lower(),
                    str(item.get("action_key") or "").lower(),
                }
                affordance_terms = {
                    "can",
                    "candidate",
                    "container",
                    "openable",
                    "success",
                    "succeeded",
                    "usable",
                    "worked",
                }
                target_overlap = len(query_terms & target_terms)
                action_overlap = len(query_terms & action_terms)
                affordance_overlap = len(query_terms & affordance_terms)
                evidence_overlap = len(query_terms & evidence_terms)
                support_count = int(item.get("support_count") or 0)
                score = (
                    95 * target_overlap
                    + 50 * action_overlap
                    + 55 * affordance_overlap
                    + 8 * evidence_overlap
                    + 9 * min(support_count, 5)
                    + 28 * float(item.get("confidence") or 0)
                    + 0.01 * int(item.get("last_step") or 0)
                )
                if query_intent == "constraint":
                    score += 125
                    if not (
                        target_overlap
                        or action_overlap
                        or affordance_overlap
                        or evidence_overlap
                    ):
                        score += 25
                elif query_intent == "habit":
                    score *= 0.20
                elif query_intent == "location":
                    score *= 0.45
                counterpart = failure_by_target.get(
                    (
                        item.get("namespace") or "",
                        item.get("action_key") or "",
                        item.get("target_key") or "",
                    )
                )
                state = interaction_state_by_target.get(
                    (
                        item.get("namespace") or "",
                        item.get("action_key") or "",
                        item.get("target_key") or "",
                    )
                )
                if counterpart and _safe_int(counterpart.get("last_step")) > _safe_int(
                    item.get("last_step")
                ):
                    score *= 0.35
                if state:
                    if state.get("current_status") == "available":
                        score *= 1.12
                    elif state.get("current_status") == "blocked":
                        score *= 0.35
                    if int(state.get("conflict_count") or 0) > 0:
                        score *= 0.90
                if (
                    query_terms
                    and not (
                        target_overlap
                        or action_overlap
                        or affordance_overlap
                        or evidence_overlap
                    )
                    and query_intent != "constraint"
                ):
                    continue
            else:
                objects = set(item.get("support_objects") or [])
                object_terms = objects | set((item.get("object_terms") or {}).keys())
                failed_targets = set(item.get("negative_targets") or [])
                target_terms = {item.get("target_key", "")} | _terms(
                    item.get("target", "")
                )
                exact_object_overlap = len(exact_object_query_terms & object_terms)
                semantic_object_overlap = len(
                    semantic_object_query_terms & object_terms
                )
                object_overlap = exact_object_overlap + semantic_object_overlap
                target_overlap = len(query_terms & target_terms)
                evidence_overlap = len(query_terms & evidence_terms)
                support_count = int(item.get("support_count") or 0)
                pair_match = pair_target_matches.get(item.get("target_key"))
                if exact_object_query_terms and has_exact_pair_match:
                    if (
                        not pair_match
                        or int(pair_match.get("exact_support_count") or 0) <= 0
                    ):
                        continue
                    if pair_match.get("superseded_by_recent_exact_match"):
                        continue
                score = (
                    125 * exact_object_overlap
                    + 38 * semantic_object_overlap
                    + 12 * target_overlap
                    + 3 * evidence_overlap
                    + 6 * support_count
                    + 4 * len(objects)
                    + 2 * len(failed_targets)
                    + 25 * float(item.get("confidence") or 0)
                )
                if pair_match:
                    score += (
                        160
                        + 135 * len(pair_match.get("matched_exact_terms") or [])
                        + 36 * len(pair_match.get("matched_semantic_terms") or [])
                        + 14 * min(int(pair_match.get("support_count") or 0), 5)
                        + 35 * float(pair_match.get("confidence") or 0)
                        + 18 * float(pair_match.get("specificity_score") or 0.0)
                        + 0.01 * max(int(pair_match.get("last_step") or 0), 0)
                    )
                elif pair_target_matches:
                    score -= 260
                if query_intent == "habit":
                    score += 100
                    if pair_match:
                        score += 80
                        if target_rank_by_key.get(item.get("target_key")) == 1:
                            score += 45 + 10 * max(
                                best_specificity - second_specificity, 0.0
                            )
                    elif exact_object_overlap:
                        score += 120
                    elif semantic_object_overlap:
                        score += 35
                    elif habit_object_overlap_available:
                        score -= 180
                elif query_intent == "location":
                    score *= 0.25
                elif query_intent == "mixed":
                    score += 30
                if (
                    query_terms
                    and not (object_overlap or target_overlap or evidence_overlap)
                    and query_intent not in {"habit", "mixed"}
                ):
                    continue
            if namespace and item.get("namespace") == namespace:
                score += 20
            ranked.append((score, item))
        ranked.sort(
            key=lambda item: (
                -item[0],
                item[1].get("target", ""),
                item[1].get("namespace", ""),
            )
        )

        results: List[dict] = []
        for score, item in ranked[:top_k]:
            if item.get("experience_type") == "observed_object_location":
                selected_target_key = item.get("target_key")
                history = sorted(
                    item.get("location_history") or [],
                    key=lambda h: (
                        h.get("target_key") == selected_target_key,
                        int(h.get("step") or -1),
                    ),
                    reverse=True,
                )
                result_object_match = _location_object_match(
                    item,
                    query_terms=query_terms,
                    semantic_object_query_terms=semantic_object_query_terms,
                )
                result_target_terms = (
                    {item.get("target_key", "")}
                    | {item.get("target_family_key", "")}
                    | _terms(item.get("target", ""))
                    | set((item.get("target_family_terms") or {}).keys())
                )
                matched_object_terms = sorted(
                    result_object_match.get("matched_terms") or []
                )
                matched_target_terms = sorted(query_terms & result_target_terms)
                location_policy = {
                    "preferred_action_types": ["Navigate", "PickUp"],
                    "preferred_object_key": item.get("object_key"),
                    "preferred_object_family_key": item.get("object_family_key"),
                    "preferred_exact_object_targets": sorted(
                        key for key in {item.get("object_key")} if key
                    ),
                    "fallback_object_families": sorted(
                        key
                        for key in {item.get("object_family_key")}
                        if key and key != item.get("object_key")
                    ),
                    "preferred_object_families": sorted(
                        key
                        for key in {
                            item.get("object_key"),
                            item.get("object_family_key"),
                        }
                        if key
                    ),
                    "preferred_location_target_key": item.get("target_key"),
                    "preferred_location_family_key": item.get("target_family_key"),
                    "preferred_location_targets": sorted(
                        key
                        for key in {
                            item.get("target_key"),
                            item.get("target_family_key"),
                        }
                        if key
                    ),
                    "relation": item.get("relation"),
                    "target_support": _ranked_location_support(
                        item.get("target_support") or {},
                        limit=5,
                        preferred_key=item.get("target_key"),
                    ),
                    "target_family_support": _ranked_location_support(
                        item.get("target_family_support") or {},
                        limit=5,
                        preferred_key=item.get("target_family_key"),
                    ),
                }
                results.append(
                    {
                        "id": item.get("id"),
                        "_layer": "experience",
                        "experience_type": "observed_object_location",
                        "namespace": item.get("namespace"),
                        "object": item.get("object"),
                        "object_key": item.get("object_key"),
                        "object_family": item.get("object_family"),
                        "object_family_key": item.get("object_family_key"),
                        "object_variants": sorted(item.get("object_variants") or []),
                        "target": item.get("target"),
                        "target_key": item.get("target_key"),
                        "target_family": item.get("target_family"),
                        "target_family_key": item.get("target_family_key"),
                        "target_variants": sorted(item.get("target_variants") or []),
                        "relation": item.get("relation"),
                        "last_step": item.get("last_step"),
                        "observation_count": item.get("observation_count"),
                        "movement_count": item.get("movement_count"),
                        "last_previous_target": item.get("last_previous_target"),
                        "last_previous_target_key": item.get(
                            "last_previous_target_key"
                        ),
                        "confidence": item.get("confidence"),
                        "matched_object_terms": matched_object_terms,
                        "matched_target_terms": matched_target_terms,
                        "object_match_level": result_object_match.get("level"),
                        "source_context_terms": _ranked_counts(
                            item.get("source_context_terms") or {},
                            limit=5,
                        ),
                        "source_rooms": _ranked_counts(
                            item.get("source_rooms") or {}, limit=5
                        ),
                        "last_room_type": item.get("last_room_type"),
                        "last_scene": item.get("last_scene"),
                        "last_session_name": item.get("last_session_name"),
                        "location_policy": location_policy,
                        "location_history": history[:5],
                        "movement_history": sorted(
                            item.get("movement_history") or [],
                            key=lambda h: int(h.get("step") or -1),
                            reverse=True,
                        )[:5],
                        "score": score,
                        "evidence": list(item.get("evidence") or [])[:4],
                        "source_event_ids": list(
                            dict.fromkeys(item.get("source_event_ids") or [])
                        ),
                    }
                )
            elif item.get("experience_type") == "interaction_state":
                results.append(
                    {
                        "id": item.get("id"),
                        "_layer": "experience",
                        "experience_type": "interaction_state",
                        "namespace": item.get("namespace"),
                        "action_type": item.get("action_type"),
                        "action_key": item.get("action_key"),
                        "target": item.get("target"),
                        "target_key": item.get("target_key"),
                        "target_family": item.get("target_family"),
                        "target_family_key": item.get("target_family_key"),
                        "target_variants": sorted(item.get("target_variants") or []),
                        "score": score,
                        "current_status": item.get("current_status"),
                        "status_reason": item.get("status_reason"),
                        "blocked_action_types": sorted(
                            item.get("blocked_action_types") or []
                        ),
                        "available_action_types": sorted(
                            item.get("available_action_types") or []
                        ),
                        "success_count": item.get("success_count"),
                        "failure_count": item.get("failure_count"),
                        "failure_modes": dict(item.get("failure_modes") or {}),
                        "conflict_count": item.get("conflict_count"),
                        "confidence": item.get("confidence"),
                        "first_step": item.get("first_step"),
                        "last_step": item.get("last_step"),
                        "last_outcome": item.get("last_outcome"),
                        "evidence": list(item.get("evidence") or [])[:4],
                        "source_event_ids": list(
                            dict.fromkeys(item.get("source_event_ids") or [])
                        ),
                    }
                )
            elif item.get("experience_type") == "interaction_failure_constraint":
                counterpart = affordance_by_target.get(
                    (
                        item.get("namespace") or "",
                        item.get("action_key") or "",
                        item.get("target_key") or "",
                    )
                )
                state = interaction_state_by_target.get(
                    (
                        item.get("namespace") or "",
                        item.get("action_key") or "",
                        item.get("target_key") or "",
                    )
                )
                results.append(
                    {
                        "id": item.get("id"),
                        "_layer": "experience",
                        "experience_type": "interaction_failure_constraint",
                        "namespace": item.get("namespace"),
                        "action_type": item.get("action_type"),
                        "target": item.get("target"),
                        "target_key": item.get("target_key"),
                        "target_family": item.get("target_family"),
                        "target_family_key": item.get("target_family_key"),
                        "target_variants": sorted(item.get("target_variants") or []),
                        "failure_modes": dict(item.get("failure_modes") or {}),
                        "score": score,
                        "support_count": item.get("support_count"),
                        "confidence": item.get("confidence"),
                        "last_step": item.get("last_step"),
                        "current_status": state.get("current_status")
                        if state
                        else "blocked",
                        "is_current_policy": bool(item.get("is_current_policy")),
                        "success_count": state.get("success_count") if state else 0,
                        "failure_count": state.get("failure_count")
                        if state
                        else item.get("support_count"),
                        "conflict_count": state.get("conflict_count") if state else 0,
                        "superseded_by_success": bool(
                            item.get("superseded_by_success")
                            or (
                                counterpart
                                and _safe_int(counterpart.get("last_step"))
                                > _safe_int(item.get("last_step"))
                            )
                        ),
                        "evidence": list(item.get("evidence") or [])[:4],
                        "source_event_ids": list(
                            dict.fromkeys(item.get("source_event_ids") or [])
                        ),
                    }
                )
            elif item.get("experience_type") == "interaction_affordance":
                counterpart = failure_by_target.get(
                    (
                        item.get("namespace") or "",
                        item.get("action_key") or "",
                        item.get("target_key") or "",
                    )
                )
                state = interaction_state_by_target.get(
                    (
                        item.get("namespace") or "",
                        item.get("action_key") or "",
                        item.get("target_key") or "",
                    )
                )
                results.append(
                    {
                        "id": item.get("id"),
                        "_layer": "experience",
                        "experience_type": "interaction_affordance",
                        "namespace": item.get("namespace"),
                        "action_type": item.get("action_type"),
                        "target": item.get("target"),
                        "target_key": item.get("target_key"),
                        "target_family": item.get("target_family"),
                        "target_family_key": item.get("target_family_key"),
                        "target_variants": sorted(item.get("target_variants") or []),
                        "score": score,
                        "support_count": item.get("support_count"),
                        "confidence": item.get("confidence"),
                        "last_step": item.get("last_step"),
                        "current_status": state.get("current_status")
                        if state
                        else "available",
                        "is_current_policy": bool(item.get("is_current_policy")),
                        "success_count": state.get("success_count")
                        if state
                        else item.get("support_count"),
                        "failure_count": state.get("failure_count") if state else 0,
                        "conflict_count": state.get("conflict_count") if state else 0,
                        "conflicted_by_failure": bool(
                            item.get("conflicted_by_failure")
                            or (
                                counterpart
                                and _safe_int(counterpart.get("last_step"))
                                > _safe_int(item.get("last_step"))
                            )
                        ),
                        "evidence": list(item.get("evidence") or [])[:4],
                        "source_event_ids": list(
                            dict.fromkeys(item.get("source_event_ids") or [])
                        ),
                    }
                )
            else:
                pair_match = pair_target_matches.get(item.get("target_key")) or {}
                target_rank = target_rank_by_key.get(item.get("target_key"))
                if target_rank == 1:
                    object_target_margin = round(
                        best_specificity - second_specificity, 4
                    )
                elif target_rank:
                    object_target_margin = round(
                        float(pair_match.get("specificity_score") or 0.0)
                        - best_specificity,
                        4,
                    )
                else:
                    object_target_margin = None
                preferred_target_families = {
                    key for key in {item.get("target_key")} if key
                }
                physical_target_families = sorted(
                    key
                    for key in (item.get("physical_target_counts") or {}).keys()
                    if key
                )
                placement_policy = {
                    "preferred_action_types": sorted(
                        item.get("preferred_action_types") or []
                    ),
                    "preferred_target_key": item.get("target_key"),
                    "preferred_physical_target_families": physical_target_families,
                    "preferred_target_families": sorted(preferred_target_families),
                    "grounding_target_families": physical_target_families,
                    "blocked_target_keys": sorted(
                        item.get("negative_target_counts") or {}
                    ),
                    "blocked_target_families": sorted(
                        item.get("blocked_target_keys") or []
                    ),
                    "support_object_categories": _ranked_counts(
                        item.get("object_category_counts") or {},
                        limit=8,
                    ),
                }
                results.append(
                    {
                        "id": item.get("id"),
                        "_layer": "experience",
                        "experience_type": "corrected_placement_habit",
                        "namespace": item.get("namespace"),
                        "target": item.get("target"),
                        "target_key": item.get("target_key"),
                        "target_variants": sorted(item.get("target_variants") or []),
                        "score": score,
                        "support_count": item.get("support_count"),
                        "confidence": item.get("confidence"),
                        "support_objects": sorted(item.get("support_objects") or []),
                        "support_object_counts": dict(
                            item.get("support_object_counts") or {}
                        ),
                        "support_object_categories": _ranked_counts(
                            item.get("object_category_counts") or {},
                            limit=8,
                        ),
                        "preferred_action_types": placement_policy[
                            "preferred_action_types"
                        ],
                        "placement_policy": placement_policy,
                        "object_target_support_count": (
                            pair_target_matches.get(item.get("target_key"), {}).get(
                                "support_count", 0
                            )
                        ),
                        "object_target_exact_support_count": (
                            pair_target_matches.get(item.get("target_key"), {}).get(
                                "exact_support_count",
                                0,
                            )
                        ),
                        "object_target_semantic_support_count": (
                            pair_target_matches.get(item.get("target_key"), {}).get(
                                "semantic_support_count",
                                0,
                            )
                        ),
                        "current_exact_object_target": bool(
                            pair_match.get("current_exact_object_target")
                        ),
                        "object_belief_status": pair_match.get("object_belief_status"),
                        "object_belief_superseded_by_target_key": pair_match.get(
                            "object_belief_superseded_by_target_key"
                        ),
                        "object_belief_max_source_authority": pair_match.get(
                            "object_belief_max_source_authority"
                        ),
                        "object_belief_last_source_authority": pair_match.get(
                            "object_belief_last_source_authority"
                        ),
                        "superseded_by_recent_exact_match": bool(
                            pair_match.get("superseded_by_recent_exact_match")
                        ),
                        "memory_scope": (
                            "object_target" if pair_match else "target_generalization"
                        ),
                        "object_target_rank": target_rank,
                        "object_target_margin": object_target_margin,
                        "competing_object_targets": target_competition[:5],
                        "matched_object_terms": sorted(
                            pair_target_matches.get(item.get("target_key"), {}).get(
                                "matched_terms", []
                            )
                        ),
                        "matched_exact_object_terms": sorted(
                            pair_target_matches.get(item.get("target_key"), {}).get(
                                "matched_exact_terms", []
                            )
                        ),
                        "matched_semantic_object_terms": sorted(
                            pair_target_matches.get(item.get("target_key"), {}).get(
                                "matched_semantic_terms", []
                            )
                        ),
                        "negative_targets": sorted(item.get("negative_targets") or []),
                        "negative_target_counts": dict(
                            item.get("negative_target_counts") or {}
                        ),
                        "physical_target_variants": sorted(
                            item.get("physical_target_variants") or []
                        ),
                        "physical_target_terms": dict(
                            item.get("physical_target_terms") or {}
                        ),
                        "physical_target_counts": dict(
                            item.get("physical_target_counts") or {}
                        ),
                        "grounding_aliases": _ranked_counts(
                            item.get("physical_target_counts") or {},
                            limit=5,
                        ),
                        "support_traces": list(item.get("support_traces") or [])[:6],
                        "support_examples": [
                            dict(example)
                            for example in (item.get("support_examples") or [])[:5]
                        ],
                        "evidence": list(item.get("evidence") or [])[:4],
                        "source_event_ids": list(
                            dict.fromkeys(item.get("source_event_ids") or [])
                        ),
                    }
                )
        return results

    def query_locations(
        self, query: str = "", top_k: int = 5, namespace: str = None
    ) -> List[dict]:
        return self.query(
            query,
            top_k=top_k,
            namespace=namespace,
            memory_types={"observed_object_location"},
        )

    def query_corrected_placements(
        self,
        query: str = "",
        top_k: int = 5,
        namespace: str = None,
    ) -> List[dict]:
        return self.query(
            query,
            top_k=top_k,
            namespace=namespace,
            memory_types={"corrected_placement_habit"},
        )

    def query_portfolio(
        self,
        query: str = "",
        top_k: int = 5,
        namespace: str = None,
    ) -> List[dict]:
        """Retrieve a typed portfolio instead of letting one memory type dominate.

        This is a retrieval strategy, not prompt shaping: each typed index is
        queried separately, then de-duplicated and filled by the global ranker.
        Mixed L2 probes often need both latest-location facts and interaction
        constraints, while L3 probes need corrected-placement habits.
        """
        if top_k <= 0:
            return []
        intent = _query_intent(query)
        budgets = {
            "habit": 1,
            "location": 1,
            "constraint": 1,
        }
        if intent == "habit":
            budgets["habit"] = max(2, top_k)
            budgets["location"] = 0
        elif intent == "location":
            budgets["location"] = max(2, top_k)
            budgets["habit"] = 0
        elif intent == "constraint":
            budgets["constraint"] = max(2, top_k)
            budgets["location"] = 0
        elif intent == "mixed":
            budgets = {"habit": 2, "location": 2, "constraint": 2}

        buckets: dict[str, list[dict]] = {
            "habit": [],
            "location": [],
            "constraint": [],
            "global": [],
        }
        seen_ids: set[str] = set()

        def add_items(bucket: str, items: list[dict]) -> None:
            for item in items:
                item_id = str(item.get("id") or "")
                if item_id and item_id in seen_ids:
                    continue
                if item_id:
                    seen_ids.add(item_id)
                buckets[bucket].append(item)

        if budgets["habit"]:
            add_items(
                "habit",
                self.query_corrected_placements(
                    query,
                    top_k=budgets["habit"],
                    namespace=namespace,
                ),
            )
        if budgets["location"]:
            add_items(
                "location",
                self.query_locations(
                    query,
                    top_k=budgets["location"],
                    namespace=namespace,
                ),
            )
        if budgets["constraint"]:
            add_items(
                "constraint",
                self.query_interaction_knowledge(
                    query,
                    top_k=budgets["constraint"],
                    namespace=namespace,
                ),
            )

        selected: list[dict] = []
        selected_ids: set[str] = set()

        def select(item: dict) -> None:
            item_id = str(item.get("id") or "")
            if item_id and item_id in selected_ids:
                return
            if item_id:
                selected_ids.add(item_id)
            selected.append(item)

        for bucket_name in ("habit", "location", "constraint"):
            bucket = sorted(
                buckets[bucket_name],
                key=lambda item: -float(item.get("score") or 0.0),
            )
            if bucket and len(selected) < top_k:
                select(bucket[0])

        if len(selected) < top_k:
            global_query_kwargs = {
                "top_k": top_k,
                "namespace": namespace,
            }
            if self.consolidated_portfolio_only:
                global_query_kwargs["memory_types"] = {
                    "corrected_placement_habit",
                    "interaction_state",
                    "observed_object_location",
                }
            add_items(
                "global",
                self.query(
                    query,
                    **global_query_kwargs,
                ),
            )

        remaining = [
            item
            for bucket_name in ("habit", "location", "constraint", "global")
            for item in buckets[bucket_name]
            if str(item.get("id") or "") not in selected_ids
        ]
        remaining.sort(key=lambda item: -float(item.get("score") or 0.0))
        for item in remaining:
            if len(selected) >= top_k:
                break
            select(item)
        return selected[:top_k]

    def query_interaction_states(
        self,
        query: str = "",
        top_k: int = 5,
        namespace: str = None,
    ) -> List[dict]:
        """Retrieve current merged interaction states.

        Unlike raw failure/affordance retrieval, this index returns the current
        policy after conflict resolution. It is the layer runtime action filters
        should consume.
        """
        return self.query(
            query,
            top_k=top_k,
            namespace=namespace,
            memory_types={"interaction_state"},
        )

    def query_interaction_failures(
        self,
        query: str = "",
        top_k: int = 5,
        namespace: str = None,
    ) -> List[dict]:
        return self.query(
            query,
            top_k=top_k,
            namespace=namespace,
            memory_types={"interaction_failure_constraint"},
        )

    def query_interaction_affordances(
        self,
        query: str = "",
        top_k: int = 5,
        namespace: str = None,
    ) -> List[dict]:
        return self.query(
            query,
            top_k=top_k,
            namespace=namespace,
            memory_types={"interaction_affordance"},
        )

    def query_interaction_knowledge(
        self,
        query: str = "",
        top_k: int = 5,
        namespace: str = None,
    ) -> List[dict]:
        states = self.query_interaction_states(
            query,
            top_k=top_k,
            namespace=namespace,
        )
        if states:
            return states[:top_k]

        failures = self.query_interaction_failures(
            query,
            top_k=top_k,
            namespace=namespace,
        )
        affordances = self.query_interaction_affordances(
            query,
            top_k=top_k,
            namespace=namespace,
        )
        if not failures:
            return affordances[:top_k]
        if not affordances:
            return failures[:top_k]

        failure_by_target = {
            (
                item.get("namespace") or "",
                str(item.get("action_type") or "").lower(),
                item.get("target_key") or "",
            ): item
            for item in failures
        }
        affordance_by_target = {
            (
                item.get("namespace") or "",
                str(item.get("action_type") or "").lower(),
                item.get("target_key") or "",
            ): item
            for item in affordances
        }

        merged: list[dict] = []
        for item in failures:
            candidate = dict(item)
            counterpart = affordance_by_target.get(
                (
                    candidate.get("namespace") or "",
                    str(candidate.get("action_type") or "").lower(),
                    candidate.get("target_key") or "",
                )
            )
            if counterpart and _safe_int(counterpart.get("last_step")) > _safe_int(
                candidate.get("last_step")
            ):
                candidate["superseded_by_success"] = True
                candidate["score"] = float(candidate.get("score") or 0.0) * 0.35
            merged.append(candidate)
        for item in affordances:
            candidate = dict(item)
            counterpart = failure_by_target.get(
                (
                    candidate.get("namespace") or "",
                    str(candidate.get("action_type") or "").lower(),
                    candidate.get("target_key") or "",
                )
            )
            if counterpart and _safe_int(counterpart.get("last_step")) > _safe_int(
                candidate.get("last_step")
            ):
                candidate["conflicted_by_failure"] = True
                candidate["score"] = float(candidate.get("score") or 0.0) * 0.35
            merged.append(candidate)

        merged.sort(key=lambda item: -float(item.get("score") or 0.0))
        selected = merged[:top_k]
        selected_types = {item.get("experience_type") for item in selected}
        if "interaction_failure_constraint" not in selected_types:
            selected[-1] = max(
                failures, key=lambda item: float(item.get("score") or 0.0)
            )
        if "interaction_affordance" not in selected_types:
            selected[-1] = max(
                affordances, key=lambda item: float(item.get("score") or 0.0)
            )
        selected.sort(key=lambda item: -float(item.get("score") or 0.0))
        return selected[:top_k]

    def infer_query_intent(self, query: str = "") -> str:
        return _query_intent(query)

    @staticmethod
    def _namespace(metadata: dict) -> str | None:
        return (
            metadata.get("memory_namespace")
            or metadata.get("source_episode_id")
            or metadata.get("household_id")
        )
