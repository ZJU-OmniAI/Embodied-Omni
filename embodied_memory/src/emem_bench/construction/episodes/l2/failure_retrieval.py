"""L2 类型2: 失败经验检索"""

import json
import os
import re
import tempfile
from pathlib import Path

from ...scene_utils import TrajectoryGenerator, get_room_type, load_scene_metadata
from ...scene_utils.procthor import is_procthor_scene
from ...schema import (
    Episode,
    MicroProbe,
    HiddenRule,
    MemoryCue,
    ExpectedAction,
    EvaluationHooks,
    EvaluationTrigger,
    DifficultyLevel,
    MemoryCueType,
    PenaltyType,
)
from ..builder import EpisodeBuilder


_PUT_COMPAT_CACHE: dict[str, list[str]] | None = None
_PLAN_EXECUTABLE_CACHE: dict[tuple[str, str, str], bool] = {}
_VALID_STORAGE_TYPES = {
    "Cabinet",
    "Fridge",
    "Microwave",
    "Safe",
    "Dresser",
    "Drawer",
}


def _load_put_compat() -> dict[str, list[str]]:
    global _PUT_COMPAT_CACHE
    if _PUT_COMPAT_CACHE is None:
        compat_path = (
            Path(__file__).resolve().parents[2] / "assets" / "put_compat_ai2thor.json"
        )
        with open(compat_path, "r", encoding="utf-8") as f:
            _PUT_COMPAT_CACHE = json.load(f)
    return _PUT_COMPAT_CACHE


def _storage_candidates(meta: dict) -> list[dict]:
    priority = {
        "Cabinet": 0,
        "Fridge": 1,
        "Microwave": 2,
        "Safe": 3,
        "Dresser": 4,
        "Drawer": 5,
    }
    return sorted(
        [
            obj
            for obj in meta["objects"]
            if obj.get("openable")
            and obj.get("receptacle")
            and not obj.get("pickupable")
            and not obj.get("isOpen")
            and obj.get("objectType") in _VALID_STORAGE_TYPES
            # ProcTHOR exposes nested drawer handles/sub-receptacles such as
            # Dresser|4|1___0. They look openable in metadata but are often not
            # reachable with the benchmark's high-level Navigate + Open API.
            and "___" not in obj.get("objectId", "")
        ],
        key=lambda obj: (
            priority.get(obj.get("objectType"), 100),
            obj.get("objectId", ""),
        ),
    )


def _metadata_by_id(meta: dict) -> dict[str, dict]:
    return {obj["objectId"]: obj for obj in meta["objects"]}


def _parent_ids_for(obj: dict, by_id: dict[str, dict]) -> set[str]:
    object_id = obj.get("objectId")
    parent_ids = set(obj.get("parentReceptacles") or [])
    if object_id:
        for parent in by_id.values():
            if object_id in (parent.get("receptacleObjectIds") or []):
                parent_ids.add(parent["objectId"])
    return parent_ids


def _blocked_by_closed_parent(obj: dict, by_id: dict[str, dict]) -> bool:
    for parent_id in _parent_ids_for(obj, by_id):
        parent = by_id.get(parent_id)
        if parent and parent.get("openable") and not parent.get("isOpen"):
            return True
    return False


def _already_in_valid_storage(obj: dict, by_id: dict[str, dict]) -> bool:
    for parent_id in _parent_ids_for(obj, by_id):
        parent = by_id.get(parent_id)
        if parent and parent.get("objectType") in _VALID_STORAGE_TYPES:
            return True
    return False


def _stash_candidates(
    meta: dict,
    storage_ids: set[str],
    safe_container: dict,
    clean_meta: dict,
    excluded_ids: set[str],
) -> list[dict]:
    compat = _load_put_compat()
    clean_by_id = _metadata_by_id(clean_meta)
    preferred_stash_types = [
        "KeyChain",
        "CreditCard",
        "Watch",
        "CellPhone",
        "Pen",
        "Pencil",
        "Knife",
        "ButterKnife",
        "Fork",
        "Spoon",
        "SaltShaker",
        "PepperShaker",
        "Spatula",
        "DishSponge",
        "SoapBottle",
    ]
    candidates = []
    for obj in meta["objects"]:
        if obj["objectId"] in excluded_ids:
            continue
        if not obj.get("pickupable"):
            continue
        if safe_container["objectType"] not in compat.get(obj.get("objectType"), []):
            continue
        clean_obj = clean_by_id.get(obj["objectId"])
        if not clean_obj or not clean_obj.get("pickupable"):
            continue
        if _already_in_valid_storage(clean_obj, clean_by_id):
            continue
        if _blocked_by_closed_parent(clean_obj, clean_by_id):
            continue
        parent_ids = set(obj.get("parentReceptacles") or [])
        if parent_ids & storage_ids:
            continue
        if _blocked_by_closed_parent(obj, _metadata_by_id(meta)):
            continue
        candidates.append(obj)
    return sorted(
        candidates,
        key=lambda obj: (
            preferred_stash_types.index(obj["objectType"])
            if obj["objectType"] in preferred_stash_types
            else len(preferred_stash_types),
            not bool(obj.get("visible")),
            obj["objectId"],
        ),
    )


def _plan_executable_in_clean_scene(
    scene: str, stash_obj: dict, safe_container: dict
) -> bool:
    key = (scene, stash_obj["objectId"], safe_container["objectId"])
    cached = _PLAN_EXECUTABLE_CACHE.get(key)
    if cached is not None:
        return cached

    with tempfile.TemporaryDirectory(prefix="l2_failure_plan_") as tmp_dir:
        gen = TrajectoryGenerator(
            scene,
            tmp_dir,
            headless=True,
            allow_implicit_navigation=False,
        )
        try:
            action_targets = [
                ("Navigate", stash_obj["objectId"]),
                ("PickUp", stash_obj["objectId"]),
                ("Navigate", safe_container["objectId"]),
                ("Open", safe_container["objectId"]),
                ("PutObject", safe_container["objectId"]),
                ("Close", safe_container["objectId"]),
            ]
            for action_type, target in action_targets:
                step = gen.execute_and_record(
                    action_type=action_type,
                    target=target,
                    nl_description=f"{action_type} {target}",
                )
                if not step.feedback.success:
                    _PLAN_EXECUTABLE_CACHE[key] = False
                    return False
            _PLAN_EXECUTABLE_CACHE[key] = True
            return True
        finally:
            gen.close()


def _select_failure_plan(
    meta: dict,
    locked_container: dict | None = None,
    clean_meta: dict | None = None,
    excluded_ids: set[str] | None = None,
    scene: str | None = None,
    locked_container_types: set[str] | None = None,
    safe_container_types: set[str] | None = None,
    locked_container_id: str | None = None,
    safe_container_id: str | None = None,
    stash_object_id: str | None = None,
    validate_clean_plan: bool = True,
):
    clean_meta = clean_meta or meta
    excluded_ids = excluded_ids or set()
    containers = _storage_candidates(meta)
    if len(containers) < 2:
        return None
    storage_ids = {container["objectId"] for container in containers}
    locked_options = [locked_container] if locked_container else containers
    for locked in locked_options:
        if not locked:
            continue
        if locked_container_id and locked.get("objectId") != locked_container_id:
            continue
        if (
            locked_container_types is not None
            and locked.get("objectType") not in locked_container_types
        ):
            continue
        for safe in containers:
            if safe["objectId"] == locked["objectId"]:
                continue
            if safe_container_id and safe.get("objectId") != safe_container_id:
                continue
            if (
                safe_container_types is not None
                and safe.get("objectType") not in safe_container_types
            ):
                continue
            for stash in _stash_candidates(
                meta, storage_ids, safe, clean_meta, excluded_ids
            ):
                if stash_object_id and stash.get("objectId") != stash_object_id:
                    continue
                if (
                    scene
                    and not is_procthor_scene(scene)
                    and validate_clean_plan
                    and not _plan_executable_in_clean_scene(scene, stash, safe)
                ):
                    continue
                return locked, safe, stash
    return None


def _record_failure_event(gen_ctx: TrajectoryGenerator, locked_container: dict):
    container_type = locked_container["objectType"]
    counter_before = gen_ctx.step_counter
    nav_step = gen_ctx.execute_and_record(
        action_type="Navigate",
        target=locked_container["objectId"],
        nl_description=f"Navigate to {container_type} for storage",
    )
    if not nav_step.feedback.success:
        gen_ctx._discard_steps([nav_step], counter_before)
        return []
    fail_step = gen_ctx.execute_and_record(
        action_type="Open",
        target=locked_container["objectId"],
        nl_description=f"Try to open {container_type}",
        simulate_failure=True,
        failure_message=(
            "Action Failed: Locked. "
            f"The {container_type} appears to be locked and cannot be opened."
        ),
        event_memory_tag="failure_locked_container",
    )
    return [nav_step, fail_step]


def _record_safe_affordance_event(
    gen_ctx: TrajectoryGenerator,
    safe_container: dict,
):
    """Record real positive evidence for the probe's selected safe container."""
    container_type = safe_container["objectType"]
    counter_before = gen_ctx.step_counter
    steps = []
    for action_type, description in (
        ("Navigate", f"Navigate to {container_type}"),
        ("Open", f"Open {container_type}"),
        ("Close", f"Close {container_type}"),
    ):
        step = gen_ctx.execute_and_record(
            action_type=action_type,
            target=safe_container["objectId"],
            nl_description=description,
        )
        steps.append(step)
        if not step.feedback.success:
            gen_ctx._discard_steps(steps, counter_before)
            return []
    return steps


def _record_fixed_reversible_noise(gen_noise: TrajectoryGenerator):
    """Record deterministic distractor actions without touching scene objects."""
    return [
        gen_noise.execute_and_record(
            action_type="RotateLeft",
            target=None,
            nl_description="Rotate left as a fixed distractor",
        ),
        gen_noise.execute_and_record(
            action_type="RotateRight",
            target=None,
            nl_description="Rotate right to restore orientation",
        ),
    ]


def _failure_probe_actions(
    *,
    stash_object_id: str,
    safe_container_id: str,
    close_after_put: bool = True,
) -> list[ExpectedAction]:
    """Build the released six-step plan or the OOD five-step variant."""
    actions = [
        ExpectedAction(action_type="Navigate", target=stash_object_id),
        ExpectedAction(action_type="PickUp", target=stash_object_id),
        ExpectedAction(action_type="Navigate", target=safe_container_id),
        ExpectedAction(action_type="Open", target=safe_container_id),
        ExpectedAction(action_type="PutObject", target=safe_container_id),
    ]
    if close_after_put:
        actions.append(ExpectedAction(action_type="Close", target=safe_container_id))
    return actions


def generate_l2_failure_retrieval(
    output_dir: str,
    context_scene: str = "FloorPlan1",
    noise_scene: str = "FloorPlan201",
    n_context_tasks: int = 3,
    n_noise_tasks: int = 5,
    seed: int = 42,
    locked_container_types: set[str] | None = None,
    safe_container_types: set[str] | None = None,
    locked_container_id: str | None = None,
    safe_container_id: str | None = None,
    stash_object_id: str | None = None,
    episode_variant: str | None = None,
    record_safe_affordance: bool = False,
    validate_clean_plan: bool = True,
    close_after_put: bool = True,
) -> Episode:
    """
    L2 类型2: 失败经验检索

    结构:
      Context Session: L1 任务 + 锁死收纳容器失败事件
      Noise Session: 无关 L1 任务干扰
      Probe: "把物体藏到收纳容器里" → 必须避开锁死的容器

    测试目标: 模型能否记住失败交互并在后续避免重复
    """
    if (
        locked_container_id
        and safe_container_id
        and locked_container_id == safe_container_id
    ):
        raise ValueError("locked_container_id and safe_container_id must differ")

    ep_id = f"ep_l2_failure_retrieval_{context_scene}_{noise_scene}"
    if episode_variant:
        variant = re.sub(r"[^A-Za-z0-9_-]+", "_", episode_variant).strip("_")
        if not variant:
            raise ValueError("episode_variant must contain an alphanumeric character")
        ep_id += f"_{variant}"
    ep_dir = os.path.join(output_dir, ep_id)

    builder = EpisodeBuilder(
        ep_id,
        "L2 Failure Experience Retrieval",
        "Test whether the model can remember a previous failed interaction with a locked storage container and avoid repeating it after distractor activity.",
        DifficultyLevel.L2,
    )
    if not close_after_put:
        builder.set_scene_metadata(
            {
                "ood_template_variant": "interaction_put_without_close",
            }
        )

    # ─── Context Session ────────────────────────
    ctx_dir = os.path.join(ep_dir, "context")
    print(f"\n  [Context] 启动 {context_scene} ...")
    gen_ctx = TrajectoryGenerator(context_scene, ctx_dir)
    initial_meta = load_scene_metadata(context_scene)

    if n_context_tasks > 0:
        ctx_steps, ctx_descs = gen_ctx.generate_l1_noise(
            n_tasks=max(1, n_context_tasks - 1), seed=seed
        )
    else:
        ctx_steps, ctx_descs = [], []
    operated_ids = {
        step.action.target
        for step in ctx_steps
        if step.action.action_type == "PickUp" and step.action.target
    }

    plan = None
    failure_steps = []
    safe_affordance_steps = []
    for locked_candidate in _storage_candidates(gen_ctx.get_metadata()):
        if (
            locked_container_id is not None
            and locked_candidate.get("objectId") != locked_container_id
        ):
            continue
        if (
            locked_container_types is not None
            and locked_candidate.get("objectType") not in locked_container_types
        ):
            continue
        candidate_plan = _select_failure_plan(
            gen_ctx.get_metadata(),
            locked_candidate,
            clean_meta=initial_meta,
            excluded_ids=operated_ids,
            scene=context_scene,
            locked_container_types=locked_container_types,
            safe_container_types=safe_container_types,
            locked_container_id=locked_container_id,
            safe_container_id=safe_container_id,
            stash_object_id=stash_object_id,
            validate_clean_plan=validate_clean_plan,
        )
        if not candidate_plan:
            continue
        steps = _record_failure_event(gen_ctx, locked_candidate)
        if steps:
            candidate_safe_steps = []
            if record_safe_affordance:
                candidate_safe_steps = _record_safe_affordance_event(
                    gen_ctx,
                    candidate_plan[1],
                )
                if not candidate_safe_steps:
                    gen_ctx.close()
                    return None
            plan = candidate_plan
            failure_steps = steps
            safe_affordance_steps = candidate_safe_steps
            break
    if not plan:
        print("[ERROR] 场景中找不到可执行的失败收纳容器事件")
        gen_ctx.close()
        return None

    locked_container, safe_container, stash_obj = plan
    ctx_steps.extend(failure_steps)
    ctx_steps.extend(safe_affordance_steps)
    locked_type = locked_container["objectType"]

    builder.add_hidden_rule(
        HiddenRule(
            rule_id="rule_locked",
            description=f"{locked_container['objectId']} is locked",
            rule_type="interaction_constraint",
            condition={"object_id": locked_container["objectId"], "action": "Open"},
            effect="Action Failed: Locked",
            related_sessions=[1],
        )
    )

    builder.add_memory_cue(
        MemoryCue(
            cue_id="cue_locked_container",
            cue_type=MemoryCueType.FAILURE_EVENT,
            description=f"Opening {locked_container['objectId']} failed",
            planted_in_session=1,
            planted_at_step=failure_steps[-1].step_id,
            object_id=locked_container["objectId"],
            tested_in_probe="probe_avoid_failure",
        )
    )

    leave_step = gen_ctx.generate_leave_room_step(get_room_type(noise_scene))
    ctx_steps.append(leave_step)
    gen_ctx.close()

    builder.add_session(
        session_name=f"{get_room_type(context_scene)} context task with failure event",
        scene=context_scene,
        trajectory_description=(
            f"Executed context tasks. During the session, opening a locked "
            f"{locked_type} failed."
            + (
                f" Opening the alternative {safe_container['objectType']} succeeded."
                if record_safe_affordance
                else ""
            )
        ),
        steps=ctx_steps,
    )

    # ─── Noise Session ──────────────────────────
    noise_dir = os.path.join(ep_dir, "noise")
    print(f"\n  [Noise] 启动 {noise_scene} ...")
    gen_noise = TrajectoryGenerator(noise_scene, noise_dir)

    enter_step = gen_noise.generate_enter_room_step(get_room_type(context_scene))
    if n_noise_tasks > 0:
        noise_steps, noise_descs = gen_noise.generate_l1_noise(
            n_tasks=n_noise_tasks,
            seed=seed + 200,
        )
    else:
        noise_descs = ["Executed a fixed reversible orientation distractor."]
        noise_steps = _record_fixed_reversible_noise(gen_noise)
    noise_steps = [enter_step] + noise_steps

    probe = MicroProbe(
        probe_id="probe_avoid_failure",
        probe_level=DifficultyLevel.L2,
        instruction=(
            f"Hide the {stash_obj['objectType']} in a concealed openable storage "
            f"container in the {get_room_type(context_scene)}."
            if close_after_put
            else (
                f"Place the {stash_obj['objectType']} inside a working openable "
                f"storage container in the {get_room_type(context_scene)}."
            )
        ),
        context=(
            f"The agent previously failed to open a locked {locked_type}; after "
            f"{len(noise_steps)} distractor steps, it must remember to avoid that container."
        ),
        scene=context_scene,
        expected_actions=_failure_probe_actions(
            stash_object_id=stash_obj["objectId"],
            safe_container_id=safe_container["objectId"],
            close_after_put=close_after_put,
        ),
        optimal_steps=6 if close_after_put else 5,
        max_steps=20,
        evaluation_hooks=EvaluationHooks(
            success_condition=f"{stash_obj['objectId']} in {safe_container['objectId']} and not in {locked_container['objectId']}",
            err_triggers=[
                EvaluationTrigger(
                    trigger_condition=f"Open({locked_container['objectId']})",
                    penalty_type=PenaltyType.CRITICAL_ERR,
                    weight=5.0,
                    description="Tried to open a known locked storage container again, indicating failure to remember the previous interaction.",
                ),
            ],
            rar_triggers=[
                EvaluationTrigger(
                    trigger_condition=f"Navigate to room other than {get_room_type(context_scene)}",
                    penalty_type=PenaltyType.BLIND_NAVIGATION,
                    weight=2.0,
                    description=f"The instruction asks for storage in the {get_room_type(context_scene)}, so the agent should not go to another room.",
                ),
            ],
        ),
    )
    gen_noise.close()

    builder.add_session(
        session_name=f"{get_room_type(noise_scene)} distractor task execution",
        scene=noise_scene,
        trajectory_description="Executed unrelated distractor tasks to create memory interference.",
        steps=noise_steps,
        micro_probe=probe,
    )

    return builder.build()
