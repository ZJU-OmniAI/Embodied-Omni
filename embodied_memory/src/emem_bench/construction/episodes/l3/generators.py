"""
L3 Episode 主生成函数。

generate_l3_by_rule:   根据单条经验规则生成一个 L3 Episode
generate_l3_all_rules: 遍历所有规则批量生成
"""

import os
import tempfile
from typing import Callable, Optional

import numpy as np

from ...scene_utils import TrajectoryGenerator, get_room_type, load_scene_metadata
from ...schema import (
    Episode,
    MacroProbe,
    SubTask,
    MemoryRequirement,
    MemoryCue,
    MemoryCueType,
    HiddenRule,
    EvaluationHooks,
    EvaluationTrigger,
    DifficultyLevel,
    PenaltyType,
    ExpectedAction,
)
from ...experience_templates import ConstraintRule
from ..builder import EpisodeBuilder
from .object_finders import (
    find_objects_by_filter,
    find_wrong_placement_target,
)
from .rule_helpers import (
    is_owner_habit_rule,
    auto_failure_message,
    room_pool_for_rule,
    first_filter_value,
    sequence_uses_slot,
)
from .action_sequences import (
    probe_solution_sequence,
    materialize_probe_solution_actions,
    find_auxiliary_object,
    macro_instruction,
)
from .object_finders import find_safe_container
from .session_builder import build_constraint_failure_session, build_noise_sessions


# ═══════════════════════════════════════════════════════════════
# 场景选择
# ═══════════════════════════════════════════════════════════════


def _scene_supports_macro(
    rule: ConstraintRule,
    scene: str,
    excluded_object_types: set[str] | None = None,
    target_probe_object_type: str | None = None,
) -> bool:
    """判断场景是否包含 MacroProbe 所需的物体（规则对象 + 目标/推荐位置）。"""
    try:
        meta = load_scene_metadata(scene)
    except Exception:
        return False

    probe_obj = _find_probe_object(
        meta,
        rule,
        excluded_object_types,
        target_probe_object_type=target_probe_object_type,
    )
    if not probe_obj:
        return False

    is_owner_habit = is_owner_habit_rule(rule)
    macro_target_objects = find_objects_by_filter(meta, rule.target_filter)
    macro_preferred_targets = (
        find_objects_by_filter(meta, rule.preferred_target_filter)
        if is_owner_habit and rule.preferred_target_filter
        else []
    )

    if is_owner_habit:
        if not macro_preferred_targets:
            return False

    sol_seq = probe_solution_sequence(rule)
    if sequence_uses_slot(sol_seq, "target_object") and not macro_target_objects:
        return False
    if sequence_uses_slot(sol_seq, "preferred_target") and not macro_preferred_targets:
        return False
    if sequence_uses_slot(sol_seq, "aux_object"):
        aux_object = find_auxiliary_object(
            meta, rule, avoid_object_id=probe_obj["objectId"]
        )
        if not aux_object:
            return False
    return True


def _select_macro_scene(
    rule: ConstraintRule,
    used_scenes: set[str],
    excluded_object_types: set[str] | None = None,
    scene_pool: list[str] | None = None,
    target_probe_object_type: str | None = None,
) -> Optional[str]:
    """从场景池中选出一个满足规则物体需求的场景。

    优先选未被失败 session 用过的场景；若找不到则退而允许复用。
    整个场景池都不满足需求时返回 None，由调用方跳过 Episode。
    """
    pool = room_pool_for_rule(rule, scene_pool=scene_pool)
    for candidate in pool:
        if candidate not in used_scenes and _scene_supports_macro(
            rule, candidate, excluded_object_types, target_probe_object_type
        ):
            return candidate
    for candidate in pool:
        if _scene_supports_macro(
            rule, candidate, excluded_object_types, target_probe_object_type
        ):
            return candidate
    return None


def _macro_scene_candidates(
    rule: ConstraintRule,
    used_scenes: set[str],
    preferred_scene: Optional[str] = None,
    excluded_object_types: set[str] | None = None,
    scene_pool: list[str] | None = None,
    target_probe_object_type: str | None = None,
) -> list[str]:
    """Return metadata-compatible macro scenes, preferring unseen scenes first.

    Strict executability is checked later by _build_macro_probe.  Keeping this
    as a candidate list lets generation continue when the first metadata-valid
    scene has unreachable receptacles in ProcTHOR.
    """
    candidates: list[str] = []

    def add(scene: Optional[str]) -> None:
        if not scene or scene in candidates:
            return
        if _scene_supports_macro(
            rule,
            scene,
            excluded_object_types,
            target_probe_object_type,
        ):
            candidates.append(scene)

    if preferred_scene not in used_scenes:
        add(preferred_scene)

    pool = room_pool_for_rule(rule, scene_pool=scene_pool)
    for scene in pool:
        if scene not in used_scenes:
            add(scene)
    for scene in pool:
        add(scene)

    max_candidates = int(os.environ.get("L3_MACRO_SCENE_ATTEMPTS", "48"))
    return candidates[:max_candidates]


# ═══════════════════════════════════════════════════════════════
# 主生成函数
# ═══════════════════════════════════════════════════════════════


def generate_l3_by_rule(
    rules: dict,
    output_dir: str,
    rule_id: str,
    failure_scenes: list[str],
    macro_scene: Optional[str] = None,
    scene_pool: list[str] | None = None,
    episode_suffix: str = "",
    n_failure_events: int = 3,
    n_noise_sessions: int = 2,
    n_noise_tasks_per_session: int = 3,
    seed: int = 42,
    target_probe_object_type: str | None = None,
    preopen_preferred_target: bool = False,
) -> Optional[Episode]:
    """根据指定的经验规则生成一个 L3 Episode。

    L3 结构：[N 个 Failure Session] → [Noise Sessions] → MacroProbe
    每个 Failure Session 在不同场景中重现同一条规则的违反事件，
    MacroProbe 在新场景中考核模型能否将这些失败泛化为隐式规则并主动规避。

    rules: {rule_id: ConstraintRule} 字典，由调用方加载后传入。
    """
    rule = rules.get(rule_id)
    if not rule:
        print(f"[ERROR] 未知的经验规则: {rule_id}")
        print(f"  可用规则: {list(rules.keys())}")
        return None

    rng = np.random.RandomState(seed)
    failure_candidates = _collect_failure_candidates(
        rule, failure_scenes, n_failure_events, rng
    )
    if not failure_candidates:
        print(f"[WARN] 所有候选场景均无法找到满足规则 {rule_id} 的对象")
        return None

    # 扫描 failure_scenes 获取规则过滤器实际能匹配到的所有 objectType（不依赖 filter.value 分词）
    all_available_types = _scan_available_object_types(rule, failure_scenes)
    used_types = {c["failure_object"]["objectType"] for c in failure_candidates}

    # MacroProbe 必须使用失败 session 中未出现的类型，单/多类型规则均适用
    # 若已耗尽所有类型，逐步缩减失败 session（最少保留 2 个）直到有剩余类型
    while not (all_available_types - used_types) and len(failure_candidates) > 2:
        failure_candidates = failure_candidates[:-1]
        used_types = {c["failure_object"]["objectType"] for c in failure_candidates}
    unused_types = all_available_types - used_types
    if not unused_types:
        print(
            f"  [SKIP] 规则 {rule_id}: 可用物体类型 {all_available_types} 全部出现在失败 session 中，"
            f"无法为 MacroProbe 保留不同类型，跳过"
        )
        return None
    print(
        f"  [泛化] 可用类型={all_available_types}, 失败用={used_types}, MacroProbe将用={unused_types}"
    )
    excluded_for_probe = used_types

    n_events = len(failure_candidates)
    print(f"\n  [规则] {rule.incompatibility_rule}")
    print(f"  [失败候选] 从 {len(failure_scenes)} 个场景中收集到 {n_events} 个失败事件")
    if rule.source_scenes:
        print(f"  [来源] 场景: {rule.source_scenes}")

    # 先在 metadata 层验证 MacroProbe 能闭环，再启动模拟器写图片，避免留下无 JSON 的半成品目录。
    actual_failure_scenes = [c["scene"] for c in failure_candidates]
    macro_probe = None
    for candidate_macro_scene in _macro_scene_candidates(
        rule,
        set(actual_failure_scenes),
        preferred_scene=macro_scene,
        excluded_object_types=excluded_for_probe,
        scene_pool=scene_pool,
        target_probe_object_type=target_probe_object_type,
    ):
        if candidate_macro_scene in actual_failure_scenes:
            print(
                f"  [INFO] macro_scene={candidate_macro_scene} 与失败场景重叠，尝试作为备选"
            )
        print(f"  [Macro-Probe] 尝试执行场景: {candidate_macro_scene}")
        macro_probe = _build_macro_probe(
            rule,
            candidate_macro_scene,
            n_events,
            excluded_object_types=excluded_for_probe,
            target_probe_object_type=target_probe_object_type,
            preopen_preferred_target=preopen_preferred_target,
        )
        if macro_probe is not None:
            break
    if macro_probe is None:
        print(f"  [SKIP] 场景池中找不到 strict 可执行的 MacroProbe 场景，跳过")
        return None
    macro_scene = macro_probe.scene

    ep_id = f"ep_l3_{rule_id}{episode_suffix}"
    ep_dir = os.path.join(output_dir, ep_id)
    builder = _init_builder(rule, rule_id, ep_id, n_events)

    if not _add_failure_sessions(builder, failure_candidates, rule, ep_dir, seed):
        print(
            f"  [SKIP] 规则 {rule_id}: 至少一个 failure session 未产生可验证的结构化 cue evidence"
        )
        return None

    # Noise Sessions：排除 failure_scenes 和 macro_scene，确保与考核场景不重叠
    print(f"\n  [Noise] 生成 {n_noise_sessions} 个噪声 Session")
    noise_sessions = build_noise_sessions(
        output_dir=ep_dir,
        n_sessions=n_noise_sessions,
        n_tasks_per_session=n_noise_tasks_per_session,
        seed=seed + 5000,
        used_scenes=set(actual_failure_scenes) | {macro_scene},
        scene_pool=scene_pool,
    )
    for i, ns in enumerate(noise_sessions):
        builder.add_session(
            session_name=f"{get_room_type(ns['scene'])} unrelated task distractor (Noise {i + 1})",
            scene=ns["scene"],
            trajectory_description=f"Executed unrelated L1 tasks as distractors: {'; '.join(ns['descriptions'])}",
            steps=ns["steps"],
        )

    builder.set_macro_probe(macro_probe)
    # 记录各阶段使用的场景，供后续分析和去重检查
    scene_metadata = {
        "failure_scenes": [fc["scene"] for fc in failure_candidates],
        "noise_scenes": [ns["scene"] for ns in noise_sessions],
        "macro_scene": macro_probe.scene,
        "rule_id": rule_id,
        "rule_family": rule.rule_family,
        "incompatibility_rule": rule.incompatibility_rule,
        "source_scenes": rule.source_scenes,
    }
    if preopen_preferred_target:
        scene_metadata["ood_template_variant"] = "owner_preopen_preferred_target"
    builder.set_scene_metadata(scene_metadata)

    return builder.build()


# ── 子函数 ───────────────────────────────────────────────────────


def _collect_failure_candidates(
    rule: ConstraintRule,
    failure_scenes: list[str],
    n_failure_events: int,
    rng: np.random.RandomState,
) -> list[dict]:
    """从候选场景中收集能够实例化规则的物体组合。

    每个场景需要找到：违规物体、交互目标，owner_habit 规则还需要推荐位置。
    随机打乱违规物体列表，避免每次都选同一个物体。
    """
    candidates = []
    selected_types: set[str] = set()  # 已选类型，优先选未出现过的类型
    for fs in failure_scenes:
        if len(candidates) >= n_failure_events:
            break
        fs_meta = load_scene_metadata(fs)
        fs_failure_objs = find_objects_by_filter(fs_meta, rule.object_filter)
        fs_preferred_objs = (
            find_objects_by_filter(fs_meta, rule.preferred_target_filter)
            if is_owner_habit_rule(rule) and rule.preferred_target_filter
            else []
        )

        if is_owner_habit_rule(rule):
            preferred_type = (
                first_filter_value(rule.preferred_target_filter.value)
                if rule.preferred_target_filter
                else ""
            )
            fs_target_objs = (
                [find_wrong_placement_target(fs_meta, preferred_type)]
                if not find_objects_by_filter(fs_meta, rule.target_filter)
                else find_objects_by_filter(fs_meta, rule.target_filter)
            )
            fs_target_objs = [o for o in fs_target_objs if o is not None]
            if not fs_failure_objs or not fs_target_objs or not fs_preferred_objs:
                print(
                    f"  [SKIP] 场景 {fs} 中找不到满足 owner_habit 的对象/错误位置/推荐位置，跳过"
                )
                continue
        else:
            fs_target_objs = find_objects_by_filter(fs_meta, rule.target_filter)
            if not fs_failure_objs or not fs_target_objs:
                print(f"  [SKIP] 场景 {fs} 中找不到满足条件的对象，跳过")
                continue

        rng.shuffle(fs_failure_objs)
        # 只选类型未在已有候选中出现过的物体；若该场景无新类型则跳过，不重复
        new_type_objs = [
            o for o in fs_failure_objs if o["objectType"] not in selected_types
        ]
        if not new_type_objs:
            print(f"  [SKIP] 场景 {fs} 中无新物体类型（已有: {selected_types}），跳过")
            continue
        chosen = new_type_objs[0]
        selected_types.add(chosen["objectType"])
        candidates.append(
            {
                "failure_object": chosen,
                "target_object": fs_target_objs[0],
                "preferred_object": fs_preferred_objs[0] if fs_preferred_objs else None,
                "scene": fs,
                "meta": fs_meta,
            }
        )
    return candidates


def _init_builder(
    rule: ConstraintRule,
    rule_id: str,
    ep_id: str,
    n_events: int,
) -> EpisodeBuilder:
    """创建 EpisodeBuilder 并注册隐式规则（测试目标）。"""
    obj_label = first_filter_value(rule.object_filter.value)
    tgt_label = (
        first_filter_value(rule.preferred_target_filter.value)
        if rule.preferred_target_filter
        else first_filter_value(rule.target_filter.value)
        if rule.target_filter
        else "Target"
    )
    builder = EpisodeBuilder(
        episode_id=ep_id,
        name=(
            f"L3-Homeowner-Habit-{obj_label}-{tgt_label}"
            if is_owner_habit_rule(rule)
            else f"L3-{obj_label}-{tgt_label}-constraint-generalization"
        ),
        description=(
            f"Test whether the model can infer the homeowner habit from repeated preference violations: '{rule.incompatibility_rule}'"
            if is_owner_habit_rule(rule)
            else f"Test whether the model can infer the rule from repeated failures: '{rule.incompatibility_rule}'"
        ),
        difficulty=DifficultyLevel.L3,
    )
    # HiddenRule 记录触发条件和预期效果，供评测框架验证模型是否归纳出了隐式约束
    builder.add_hidden_rule(
        HiddenRule(
            rule_id=f"rule_{rule_id}",
            description=rule.incompatibility_rule,
            rule_type=rule.rule_family,
            condition={
                "rule_family": rule.rule_family,
                rule.object_filter.property_key: rule.object_filter.value,
                "action": "PutObject",
                **(
                    {rule.target_filter.property_key: rule.target_filter.value}
                    if rule.target_filter
                    else {}
                ),
                **(
                    {"preferred_target": rule.preferred_target_filter.value}
                    if rule.preferred_target_filter
                    else {}
                ),
            },
            effect=auto_failure_message(rule),
            related_sessions=list(range(1, n_events + 1)),
        )
    )
    return builder


def _add_failure_sessions(
    builder: EpisodeBuilder,
    failure_candidates: list[dict],
    rule: ConstraintRule,
    ep_dir: str,
    seed: int,
) -> bool:
    """为每个失败候选构建 Session 并写入 builder。

    每个 Session 在独立场景中重现规则违反事件，并附上 MemoryCue 供 MacroProbe 回忆。
    safe_obj 是备用安全容器，供 physical_constraint 规则展示正确替代动作。
    """
    for i, candidate in enumerate(failure_candidates):
        failure_obj = candidate["failure_object"]
        sess_target = candidate["target_object"]
        preferred_obj = candidate["preferred_object"]
        curr_scene = candidate["scene"]
        curr_meta = candidate["meta"]
        session_idx = i + 1
        print(
            f"\n  [Session {session_idx}] 场景={curr_scene}, 失败事件: {failure_obj['objectType']} + {sess_target['objectType']}"
        )

        session_dir = os.path.join(ep_dir, f"session_{session_idx}")
        gen = TrajectoryGenerator(curr_scene, session_dir)
        safe_obj = find_safe_container(
            curr_meta,
            failure_object_type=failure_obj["objectType"],
            avoid_object_id=failure_obj["objectId"],
        )

        steps, traj_desc = build_constraint_failure_session(
            gen=gen,
            session_idx=session_idx,
            rule=rule,
            failure_object=failure_obj,
            target_object=sess_target,
            safe_object=safe_obj,
            preferred_object=preferred_obj,
            scene=curr_scene,
            seed=seed + session_idx * 100,
        )

        if not steps:
            print(f"  [WARN] Session {session_idx} 生成失败，跳过")
            gen.close()
            return False

        event_tag = f"failure_{session_idx}"
        if not failure_cue_evidence_present(
            steps,
            object_id=failure_obj["objectId"],
            target_id=sess_target["objectId"],
            event_tag=event_tag,
        ):
            print(
                f"  [WARN] Session {session_idx} 缺少结构化 cue evidence: "
                f"object={failure_obj['objectId']}, target={sess_target['objectId']}, "
                f"event_tag={event_tag}"
            )
            gen.close()
            return False

        builder.add_session(
            session_name=f"{get_room_type(curr_scene)} task execution with failure event (Session {session_idx})",
            scene=curr_scene,
            trajectory_description=traj_desc,
            steps=steps,
        )
        builder.add_memory_cue(
            MemoryCue(
                cue_id=f"cue_failure_{session_idx}",
                cue_type=MemoryCueType.FAILURE_EVENT,
                description=(
                    f"Tried placing {failure_obj['objectType']} on {sess_target['objectType']} "
                    "and the placement was corrected by human intervention"
                    if is_owner_habit_rule(rule)
                    else f"Tried interacting {failure_obj['objectType']} with {sess_target['objectType']} and failed"
                ),
                planted_in_session=session_idx,
                planted_at_step=-1,
                object_id=failure_obj["objectId"],
                tested_in_probe="macro_probe_final",
            )
        )
        gen.close()
    return True


def _record_field(record, field: str, default=None):
    if isinstance(record, dict):
        return record.get(field, default)
    return getattr(record, field, default)


def failure_cue_evidence_present(
    steps: list,
    *,
    object_id: str,
    target_id: str,
    event_tag: str,
) -> bool:
    """Require a cue to be grounded in the serialized trajectory.

    A valid L3 failure cue must expose the exact task object somewhere in the
    structured observation/action state and must contain the tagged failed
    placement with that object still held and the declared target selected.
    Natural-language descriptions alone are intentionally insufficient.
    """
    object_observed = False
    tagged_failure_observed = False
    for step in steps:
        visible_objects = _record_field(step, "visible_objects", []) or []
        action = _record_field(step, "action", None)
        agent_state = _record_field(step, "agent_state", None)
        feedback = _record_field(step, "feedback", None)

        structured_object_ids = {
            _record_field(obj, "object_id")
            for obj in visible_objects
            if _record_field(obj, "object_id")
        }
        structured_object_ids.update(
            value
            for value in (
                _record_field(action, "target"),
                _record_field(action, "instrument"),
                _record_field(agent_state, "holding"),
            )
            if value
        )
        if object_id in structured_object_ids:
            object_observed = True

        if (
            _record_field(feedback, "event_memory_tag") == event_tag
            and _record_field(action, "target") == target_id
            and _record_field(agent_state, "holding") == object_id
        ):
            tagged_failure_observed = True

    return object_observed and tagged_failure_observed


def _resolve_macro_scene(
    rule: ConstraintRule,
    failure_scenes: list[str],
    macro_scene: Optional[str],
    excluded_object_types: set[str] | None = None,
    scene_pool: list[str] | None = None,
) -> Optional[str]:
    """确定 MacroProbe 的执行场景，优先不复用失败场景。

    找不到任何满足需求的场景时返回 None，由调用方跳过 Episode。
    """
    used_scenes = set(failure_scenes)
    if (
        macro_scene is not None
        and macro_scene not in used_scenes
        and _scene_supports_macro(rule, macro_scene, excluded_object_types)
    ):
        print(f"  [Macro-Probe] 执行场景: {macro_scene}")
        return macro_scene

    if macro_scene in used_scenes:
        print(f"  [INFO] macro_scene={macro_scene} 与失败场景重叠，重新选择")
    elif macro_scene is not None:
        print(f"  [INFO] macro_scene={macro_scene} 无法满足 MacroProbe 需求，重新选择")

    macro_scene = _select_macro_scene(
        rule,
        used_scenes,
        excluded_object_types=excluded_object_types,
        scene_pool=scene_pool,
    )
    if macro_scene is None:
        print(f"  [SKIP] 场景池中找不到满足需求的 MacroProbe 场景，跳过")
        return None
    print(f"  [Macro-Probe] 执行场景: {macro_scene}")
    return macro_scene


def _scan_available_object_types(rule: ConstraintRule, scenes: list[str]) -> set[str]:
    """扫描 scenes 中所有匹配规则过滤器的物体，收集其 objectType 集合。

    用于准确判断规则是单类型还是多类型，不依赖 filter.value 的字符串分词。
    """
    types: set[str] = set()
    for scene in scenes:
        try:
            meta = load_scene_metadata(scene)
        except Exception:
            continue
        for obj in find_objects_by_filter(meta, rule.object_filter):
            types.add(obj["objectType"])
    return types


def _split_filter_values(value: str) -> set[str]:
    return {v.strip() for v in str(value).split("|") if v.strip()}


def _build_scene_type_index(scene_pool: list[str]) -> dict[str, dict[str, set[str]]]:
    """Build a cheap per-scene type index for fast ProcTHOR candidate filtering."""
    index: dict[str, dict[str, set[str]]] = {}
    for scene in scene_pool:
        try:
            meta = load_scene_metadata(scene)
        except Exception:
            continue
        pickup_types: set[str] = set()
        receptacle_types: set[str] = set()
        for obj in meta.get("objects", []):
            obj_type = obj.get("objectType")
            if not obj_type:
                continue
            if obj.get("pickupable"):
                pickup_types.add(obj_type)
            if obj.get("receptacle") and obj_type != "Floor":
                receptacle_types.add(obj_type)
        index[scene] = {
            "pickup_types": pickup_types,
            "receptacle_types": receptacle_types,
        }
    return index


def _prefilter_failure_scene_pool(
    rule: ConstraintRule,
    scene_pool: list[str],
    scene_type_index: dict[str, dict[str, set[str]]],
) -> list[str]:
    """Drop scenes that cannot possibly instantiate this rule.

    This avoids repeatedly scanning thousands of ProcTHOR houses that do not
    contain the rule object or preferred receptacle.  The precise object-level
    checks still happen later in _collect_failure_candidates.
    """
    if not scene_type_index:
        return scene_pool
    if rule.object_filter.property_key != "objectType":
        return scene_pool

    object_types = _split_filter_values(rule.object_filter.value)
    if not object_types:
        return []

    if is_owner_habit_rule(rule):
        if not rule.preferred_target_filter:
            return []
        preferred_types = _split_filter_values(rule.preferred_target_filter.value)
        if not preferred_types:
            return []
        filtered = []
        for scene in scene_pool:
            features = scene_type_index.get(scene)
            if not features:
                continue
            pickup_types = features["pickup_types"]
            receptacle_types = features["receptacle_types"]
            if not (pickup_types & object_types):
                continue
            if not (receptacle_types & preferred_types):
                continue
            if not (receptacle_types - preferred_types):
                continue
            filtered.append(scene)
        return filtered

    if not rule.target_filter:
        return []
    target_types = _split_filter_values(rule.target_filter.value)
    filtered = []
    for scene in scene_pool:
        features = scene_type_index.get(scene)
        if not features:
            continue
        if (
            features["pickup_types"] & object_types
            and features["receptacle_types"] & target_types
        ):
            filtered.append(scene)
    return filtered


def _object_type_from_id(object_id: str | None) -> str:
    if not object_id:
        return ""
    parts = str(object_id).split("|")
    if parts and parts[-1] in {"BathtubBasin", "SinkBasin"}:
        return parts[-1]
    return parts[0] if parts else ""


def _parent_types(obj: dict) -> set[str]:
    return {
        _object_type_from_id(parent_id)
        for parent_id in (obj.get("parentReceptacles") or [])
        if parent_id
    }


def _rank_receptacle_candidate(obj: dict) -> tuple:
    """Prefer target instances that are more likely to accept placement."""
    occupied_count = len(obj.get("receptacleObjectIds") or [])
    return (
        not bool(obj.get("visible")),
        not bool(obj.get("receptacle")),
        occupied_count,
        obj.get("objectId", ""),
    )


def _sort_receptacle_candidates(objects: list[dict]) -> list[dict]:
    return sorted(objects, key=_rank_receptacle_candidate)


def _probe_object_candidates(
    meta: dict,
    rule: ConstraintRule,
    excluded_types: set[str] | None,
    target_probe_object_type: str | None = None,
) -> list[dict]:
    """返回 MacroProbe 可用规则物体候选，优先选择严格动作更可能可执行的实例。

    excluded_types 非空时：只返回类型不在其中的物体（多类型规则的泛化要求）；
    找不到时返回空列表（不回落），由调用方尝试其他场景或跳过 Episode。
    """
    all_objs = find_objects_by_filter(meta, rule.object_filter)
    if not all_objs:
        return []
    if is_owner_habit_rule(rule) and rule.preferred_target_filter:
        preferred_types = {
            t.strip()
            for t in str(rule.preferred_target_filter.value).split("|")
            if t.strip()
        }
        outside_preferred = [
            obj for obj in all_objs if not (_parent_types(obj) & preferred_types)
        ]
        if outside_preferred:
            all_objs = outside_preferred
    if excluded_types:
        all_objs = [o for o in all_objs if o["objectType"] not in excluded_types]
    if target_probe_object_type:
        all_objs = [
            obj for obj in all_objs if obj.get("objectType") == target_probe_object_type
        ]
    return sorted(
        all_objs,
        key=lambda obj: (
            not bool(obj.get("visible")),
            bool(_parent_types(obj)),
            obj.get("objectId", ""),
        ),
    )


def _find_probe_object(
    meta: dict,
    rule: ConstraintRule,
    excluded_types: set[str] | None,
    target_probe_object_type: str | None = None,
) -> Optional[dict]:
    candidates = _probe_object_candidates(
        meta,
        rule,
        excluded_types,
        target_probe_object_type=target_probe_object_type,
    )
    return candidates[0] if candidates else None


def _expected_action_field(action, field: str):
    if isinstance(action, dict):
        return action.get(field)
    return getattr(action, field, None)


def _preopen_preferred_actions(
    expected_actions: list[ExpectedAction],
    preferred_target_id: str,
) -> Optional[list[ExpectedAction]]:
    """Move opening an owner-habit destination before task-object pickup.

    The OOD variant is deliberately defined only for the standard openable
    owner-habit path so it cannot silently relabel a different action plan.
    """
    if len(expected_actions) != 5:
        return None

    navigate_object, pickup_object, navigate_preferred, open_preferred, put_object = (
        expected_actions
    )
    if [action.action_type for action in expected_actions] != [
        "Navigate",
        "PickUp",
        "Navigate",
        "Open",
        "PutObject",
    ]:
        return None
    if not navigate_object.target or pickup_object.target != navigate_object.target:
        return None
    if any(
        action.target != preferred_target_id
        for action in (navigate_preferred, open_preferred, put_object)
    ):
        return None

    return [
        ExpectedAction(action_type="Navigate", target=preferred_target_id),
        ExpectedAction(action_type="Open", target=preferred_target_id),
        ExpectedAction(action_type="Navigate", target=navigate_object.target),
        ExpectedAction(action_type="PickUp", target=pickup_object.target),
        ExpectedAction(action_type="Navigate", target=preferred_target_id),
        ExpectedAction(
            action_type="PutObject",
            target=preferred_target_id,
            object_held=put_object.object_held,
        ),
    ]


def _strict_expected_actions_executable(scene: str, expected_actions: list) -> bool:
    """Check that the macro solution can run under evaluation-style strict actions."""
    if os.environ.get("L3_STRICT_MACRO_VALIDATE", "1") == "0":
        return True

    with tempfile.TemporaryDirectory(prefix="l3_strict_probe_") as tmpdir:
        gen = TrajectoryGenerator(
            scene,
            tmpdir,
            width=int(os.environ.get("L3_STRICT_VALIDATE_RESOLUTION", "300")),
            height=int(os.environ.get("L3_STRICT_VALIDATE_RESOLUTION", "300")),
            allow_implicit_navigation=False,
        )
        try:
            for action in expected_actions:
                action_type = _expected_action_field(action, "action_type")
                target = _expected_action_field(action, "target")
                instrument = _expected_action_field(action, "object_held")
                try:
                    if action_type == "Navigate":
                        success = gen.navigate_to_object(target)
                    elif action_type == "PickUp":
                        success = gen.pickup_object(target)
                    elif action_type == "Open":
                        success = gen.open_object(target)
                    elif action_type == "Close":
                        success = gen.close_object(target)
                    elif action_type == "PutObject":
                        success = gen.put_object(target)
                    elif action_type == "TransferContents":
                        success = gen.transfer_contents(instrument, target)
                    else:
                        success = False
                except Exception as exc:
                    print(
                        f"[WARN] strict macro validation exception in {scene}: {type(exc).__name__}: {exc}"
                    )
                    return False
                if not success:
                    return False
            return True
        finally:
            gen.close()


def _build_macro_probe(
    rule: ConstraintRule,
    macro_scene: str,
    n_events: int,
    excluded_object_types: set[str] | None = None,
    target_probe_object_type: str | None = None,
    preopen_preferred_target: bool = False,
) -> Optional[MacroProbe]:
    """在 macro_scene 中实例化 MacroProbe。

    从场景元数据中查找所需物体，校验正确解答路径可达，构建评测钩子。
    任意必需物体缺失时返回 None。
    """
    meta_macro = load_scene_metadata(macro_scene)
    is_owner_habit = is_owner_habit_rule(rule)

    gen_obj_candidates = _probe_object_candidates(
        meta_macro,
        rule,
        excluded_object_types,
        target_probe_object_type=target_probe_object_type,
    )
    if not gen_obj_candidates:
        print(
            f"[WARN] MacroProbe 场景 {macro_scene} 中找不到合适的规则对象，无法生成任务"
        )
        return None

    macro_target_objects = _sort_receptacle_candidates(
        find_objects_by_filter(meta_macro, rule.target_filter)
    )
    # owner_habit 规则：preferred_targets 是屋主期望的正确摆放位置（答案）
    macro_preferred_targets = (
        _sort_receptacle_candidates(
            find_objects_by_filter(meta_macro, rule.preferred_target_filter)
        )
        if is_owner_habit and rule.preferred_target_filter
        else []
    )
    if is_owner_habit and not macro_preferred_targets:
        print(f"[WARN] MacroProbe 场景 {macro_scene} 中找不到 owner_habit 的推荐位置")
        return None

    # 校验正确解答序列所需的物体在场景中都能找到
    sol_seq = probe_solution_sequence(rule)
    if sequence_uses_slot(sol_seq, "target_object") and not macro_target_objects:
        print(f"[WARN] MacroProbe 场景 {macro_scene} 中找不到目标对象，无法生成任务")
        return None
    if sequence_uses_slot(sol_seq, "preferred_target") and not macro_preferred_targets:
        print(f"[WARN] MacroProbe 场景 {macro_scene} 中找不到推荐位置，无法生成任务")
        return None
    target_candidates = (
        macro_target_objects
        if sequence_uses_slot(sol_seq, "target_object")
        else ([macro_target_objects[0]] if macro_target_objects else [None])
    )
    preferred_candidates = (
        macro_preferred_targets
        if sequence_uses_slot(sol_seq, "preferred_target")
        else [None]
    )
    max_probe_objects = int(os.environ.get("L3_STRICT_VALIDATE_MAX_OBJECTS", "8"))
    max_targets = int(os.environ.get("L3_STRICT_VALIDATE_MAX_TARGETS", "64"))

    selected = None
    for gen_obj in gen_obj_candidates[:max_probe_objects]:
        aux_object = find_auxiliary_object(
            meta_macro, rule, avoid_object_id=gen_obj["objectId"]
        )
        if sequence_uses_slot(sol_seq, "aux_object") and not aux_object:
            continue
        for macro_target in target_candidates[:max_targets]:
            for preferred_tgt in preferred_candidates[:max_targets]:
                if preopen_preferred_target and (
                    not preferred_tgt or not preferred_tgt.get("openable")
                ):
                    continue
                expected_actions = materialize_probe_solution_actions(
                    rule=rule,
                    task_object=gen_obj,
                    target_object=macro_target,
                    aux_object=aux_object,
                    preferred_object=preferred_tgt,
                )
                if not expected_actions:
                    continue
                if preopen_preferred_target:
                    expected_actions = _preopen_preferred_actions(
                        expected_actions,
                        preferred_tgt["objectId"],
                    )
                    if expected_actions is None:
                        continue
                if _strict_expected_actions_executable(macro_scene, expected_actions):
                    selected = (
                        gen_obj,
                        macro_target,
                        preferred_tgt,
                        aux_object,
                        expected_actions,
                    )
                    break
            if selected:
                break
        if selected:
            break

    if not selected:
        print(
            f"[WARN] MacroProbe 场景 {macro_scene} 中找不到 strict 可执行的正确解答路径"
        )
        return None

    gen_obj, macro_target, preferred_tgt, aux_object, expected_actions = selected

    probe_instruction = macro_instruction(
        rule, task_object=gen_obj, target_object=macro_target
    )
    print(f"\n  [Macro-Probe] 指令: {probe_instruction}")

    err_triggers = _build_err_triggers(
        rule, gen_obj, macro_target, macro_target_objects, is_owner_habit
    )

    return MacroProbe(
        probe_id="macro_probe_final",
        probe_level=DifficultyLevel.L3,
        instruction=probe_instruction,
        scene=macro_scene,
        # required_memory_integration 声明此题需要整合哪些 session 的失败记忆
        required_memory_integration=[
            MemoryRequirement(
                memory_type="event_generalization",
                source_sessions=list(range(1, n_events + 1)),
                source_events=[f"failure_{i}" for i in range(1, n_events + 1)],
                expected_inference=rule.incompatibility_rule,
            ),
        ],
        sub_tasks=[
            SubTask(
                sub_task_id="subtask_generalize",
                instruction="Complete the current task and use the implicit experience generalized from previous failures to avoid the critical mistake.",
                expected_actions=expected_actions,
                evaluation_hooks=EvaluationHooks(
                    success_condition=(
                        "Choose the location that matches the homeowner's implicit habit."
                        if is_owner_habit
                        else "Complete the task without directly triggering the known unsafe interaction."
                    ),
                    err_triggers=err_triggers,
                    rar_triggers=[
                        EvaluationTrigger(
                            trigger_condition="Navigate to room other than current scene",
                            penalty_type=PenaltyType.BLIND_NAVIGATION,
                            weight=2.0,
                            description="Do not leave the current room to search elsewhere.",
                        ),
                    ],
                ),
            ),
        ],
        max_steps=60,
        scoring={"lambda_rar": 5, "alpha": 0.5, "beta": 0.5},
    )


def generate_l3_all_rules(
    rules: dict,
    output_dir: str,
    n_noise_sessions: int = 2,
    n_noise_tasks_per_session: int = 3,
    seed: int = 42,
    scene_pool: list[str] | None = None,
    variants_per_rule: int = 1,
    on_episode: Callable[[Episode], None] | None = None,
) -> list[Episode]:
    """遍历 rules 中所有经验规则，生成对应的 L3 Episode。

    rules: {rule_id: ConstraintRule} 字典，由调用方加载后传入。
    """
    episodes = []
    selected_rules = list(rules.items())
    used_scenes_by_rule: dict[str, set[str]] = {}
    scene_type_index = _build_scene_type_index(scene_pool) if scene_pool else {}
    if scene_pool:
        print(
            f"[L3] 已建立场景类型索引: {len(scene_type_index)}/{len(scene_pool)} 个场景"
        )

    variants_per_rule = max(1, variants_per_rule)
    for i, (rule_id, rule) in enumerate(selected_rules):
        used_scenes_by_rule.setdefault(rule_id, set())
        rule_base_scene_pool = list(scene_pool or rule.source_scenes)
        if scene_pool:
            before_filter = len(rule_base_scene_pool)
            rule_base_scene_pool = _prefilter_failure_scene_pool(
                rule,
                rule_base_scene_pool,
                scene_type_index,
            )
            print(
                f"[L3] {rule_id}: 候选场景预筛 "
                f"{before_filter} → {len(rule_base_scene_pool)}"
            )
        for variant_idx in range(variants_per_rule):
            suffix = f"_v{variant_idx + 1:03d}" if variants_per_rule > 1 else ""
            ep_id = f"ep_l3_{rule_id}{suffix}"
            output_path = os.path.join(output_dir, ep_id, f"{ep_id}.json")
            if os.path.exists(output_path):
                print(f"[SKIP] {rule_id}{suffix} 已存在，跳过")
                continue

            base_scene_pool = list(rule_base_scene_pool)
            if scene_pool:
                min_needed_scenes = 3 + n_noise_sessions + 1
                low_overlap_pool = [
                    scene
                    for scene in base_scene_pool
                    if scene not in used_scenes_by_rule[rule_id]
                ]
                if len(low_overlap_pool) >= min_needed_scenes:
                    base_scene_pool = low_overlap_pool
            failure_scene_list = list(base_scene_pool)
            if not failure_scene_list:
                print(f"[SKIP] 规则 {rule_id}{suffix}: 预筛后无候选场景，跳过")
                continue

            # 每条规则/variant 用不同偏移量，保证随机序列不重叠。
            variant_seed = seed + i * 1000 + variant_idx * 100000
            rng = np.random.RandomState(variant_seed)
            rng.shuffle(failure_scene_list)
            initial_used_scenes = (
                set(failure_scene_list)
                if scene_pool is None
                else set(used_scenes_by_rule[rule_id])
            )
            macro_scene = _select_macro_scene(
                rule,
                initial_used_scenes,
                scene_pool=failure_scene_list if scene_pool else None,
            )

            print(f"\n{'=' * 60}")
            print(
                f"生成 L3 Episode: {rule_id}{suffix} (规则: {first_filter_value(rule.object_filter.value)})"
            )
            print(f"  候选场景数: {len(failure_scene_list)}")
            print(f"  MacroProbe场景: {macro_scene}")
            print(f"  互斥规则: {rule.incompatibility_rule}")
            print(f"{'=' * 60}")

            ep = generate_l3_by_rule(
                rules=rules,
                output_dir=output_dir,
                rule_id=rule_id,
                failure_scenes=failure_scene_list,
                macro_scene=macro_scene,
                scene_pool=failure_scene_list if scene_pool else None,
                episode_suffix=suffix,
                n_failure_events=3,
                n_noise_sessions=n_noise_sessions,
                n_noise_tasks_per_session=n_noise_tasks_per_session,
                seed=variant_seed,
            )

            if ep:
                episodes.append(ep)
                scene_metadata = ep.scene_metadata or {}
                for scene in scene_metadata.get("failure_scenes", []):
                    used_scenes_by_rule[rule_id].add(scene)
                for scene in scene_metadata.get("noise_scenes", []):
                    used_scenes_by_rule[rule_id].add(scene)
                if scene_metadata.get("macro_scene"):
                    used_scenes_by_rule[rule_id].add(scene_metadata["macro_scene"])
                if on_episode:
                    on_episode(ep)
                print(f"  [OK] Episode: {ep.episode_name}")
            else:
                print(f"  [FAIL] Episode 生成失败")

    return episodes


def _build_err_triggers(
    rule: ConstraintRule,
    gen_obj: Optional[dict],
    macro_target: Optional[dict],
    macro_target_objects: list[dict],
    is_owner_habit: bool,
) -> list[EvaluationTrigger]:
    """构建 MacroProbe 的错误惩罚触发条件。

    owner_habit：放到任何"错误位置"都算违规，需枚举所有 macro_target_objects。
    physical_constraint：直接将违规物体放入目标容器，或未经转移就放入，均算违规。
    """
    err_desc = f"Rule was not applied: {rule.incompatibility_rule}"
    triggers = []

    if is_owner_habit and macro_target_objects:
        # 屋主习惯规则：所有"习惯上不该放"的位置都是错误目标
        for tgt in macro_target_objects:
            triggers.append(
                EvaluationTrigger(
                    trigger_condition=f"PutObject({tgt['objectId']})",
                    penalty_type=PenaltyType.CRITICAL_ERR,
                    weight=5.0,
                    description=err_desc,
                )
            )
    elif macro_target:
        # 物理约束规则：直接交互或未做安全中转就放入目标均触发惩罚
        triggers.append(
            EvaluationTrigger(
                trigger_condition=(
                    f"DirectUnsafeInteraction(task={gen_obj['objectId'] if gen_obj else 'unknown'}, "
                    f"target={macro_target['objectId']})"
                ),
                penalty_type=PenaltyType.CRITICAL_ERR,
                weight=5.0,
                description=err_desc,
            )
        )
        triggers.append(
            EvaluationTrigger(
                trigger_condition=f"PutObject({macro_target['objectId']}) before TransferContents",
                penalty_type=PenaltyType.CRITICAL_ERR,
                weight=5.0,
                description=err_desc,
            )
        )
    else:
        triggers.append(
            EvaluationTrigger(
                trigger_condition="Directly tried to interact the object with the target without safety handling",
                penalty_type=PenaltyType.CRITICAL_ERR,
                weight=5.0,
                description=err_desc,
            )
        )
    return triggers
