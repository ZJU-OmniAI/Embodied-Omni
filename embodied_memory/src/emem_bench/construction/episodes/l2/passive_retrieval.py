"""L2 类型1: 被动观察检索"""

import os
import re
import tempfile

from ...scene_utils import TrajectoryGenerator, get_room_type, load_scene_metadata
from ...scene_utils.procthor import is_procthor_scene
from ...schema import (
    Episode,
    MicroProbe,
    MemoryCue,
    MemoryCueExposure,
    ExpectedAction,
    EvaluationHooks,
    EvaluationTrigger,
    DifficultyLevel,
    MemoryCueType,
    PenaltyType,
)
from ..builder import EpisodeBuilder


_RETRIEVAL_PLAN_CACHE: dict[tuple[str, str, str, bool], bool] = {}


def _is_floor_id(object_id: str | None) -> bool:
    return bool(object_id) and (object_id == "Floor" or object_id.startswith("Floor|"))


def _metadata_parent(meta: dict, object_id: str) -> str | None:
    obj = next((o for o in meta["objects"] if o["objectId"] == object_id), None)
    parent_ids = (obj or {}).get("parentReceptacles") or []
    for parent_id in parent_ids:
        if not _is_floor_id(parent_id):
            return parent_id
    return None


def _has_clear_retrieval_parent(parent_id: str | None) -> bool:
    return bool(parent_id) and not _is_floor_id(parent_id)


def _targeted_observation_candidate(
    meta: dict,
    *,
    operated_ids: set[str],
    parent_types: set[str],
    object_types: set[str] | None = None,
    target_parent_id: str | None = None,
    target_object_id: str | None = None,
) -> tuple[dict, dict] | None:
    by_id = {obj["objectId"]: obj for obj in meta["objects"]}
    candidates = []
    for obj in meta["objects"]:
        if obj.get("objectId") in operated_ids or not obj.get("pickupable"):
            continue
        if target_object_id and obj.get("objectId") != target_object_id:
            continue
        if object_types is not None and obj.get("objectType") not in object_types:
            continue
        parent_id = _metadata_parent(meta, obj["objectId"])
        if target_parent_id and parent_id != target_parent_id:
            continue
        parent = by_id.get(parent_id)
        if not parent or parent.get("objectType") not in parent_types:
            continue
        if not parent.get("openable") or not parent.get("receptacle"):
            continue
        candidates.append((obj, parent))
    return min(
        candidates,
        key=lambda pair: (pair[0].get("objectType", ""), pair[0]["objectId"]),
        default=None,
    )


def _record_targeted_observation(
    gen_ctx: TrajectoryGenerator,
    target_obj: dict,
    parent_obj: dict,
):
    counter_before = gen_ctx.step_counter
    steps = []

    def execute(action_type: str, target: str, description: str):
        step = gen_ctx.execute_and_record(
            action_type=action_type,
            target=target,
            nl_description=description,
        )
        steps.append(step)
        return step

    parent_type = parent_obj["objectType"]
    object_type = target_obj["objectType"]
    nav_step = execute(
        "Navigate",
        parent_obj["objectId"],
        f"Navigate to {parent_type}",
    )
    if not nav_step.feedback.success:
        gen_ctx._discard_steps(steps, counter_before)
        return []

    current_parent = next(
        (
            obj
            for obj in gen_ctx.get_metadata()["objects"]
            if obj["objectId"] == parent_obj["objectId"]
        ),
        parent_obj,
    )
    if current_parent.get("isOpen"):
        close_step = execute(
            "Close",
            parent_obj["objectId"],
            f"Close {parent_type}",
        )
        if not close_step.feedback.success:
            gen_ctx._discard_steps(steps, counter_before)
            return []

    open_step = execute(
        "Open",
        parent_obj["objectId"],
        f"Open {parent_type}",
    )
    if not open_step.feedback.success:
        gen_ctx._discard_steps(steps, counter_before)
        return []

    target_visible = any(
        visible.object_id == target_obj["objectId"]
        for visible in open_step.visible_objects
    )
    if not target_visible:
        view_step = execute(
            "Navigate",
            target_obj["objectId"],
            f"Navigate to {object_type}",
        )
        if not view_step.feedback.success:
            gen_ctx._discard_steps(steps, counter_before)
            return []
        target_visible = any(
            visible.object_id == target_obj["objectId"]
            for visible in view_step.visible_objects
        )

    close_step = execute(
        "Close",
        parent_obj["objectId"],
        f"Close {parent_type}",
    )
    if not close_step.feedback.success or not target_visible:
        gen_ctx._discard_steps(steps, counter_before)
        return []
    return steps


def _retrieval_plan_executable(
    scene: str,
    object_id: str,
    parent_id: str,
    parent_openable: bool,
) -> bool:
    key = (scene, object_id, parent_id, parent_openable)
    cached = _RETRIEVAL_PLAN_CACHE.get(key)
    if cached is not None:
        return cached

    with tempfile.TemporaryDirectory(prefix="l2_passive_plan_") as tmp_dir:
        gen = TrajectoryGenerator(
            scene,
            tmp_dir,
            headless=True,
            allow_implicit_navigation=False,
        )
        try:
            actions = [("Navigate", parent_id)]
            if parent_openable:
                actions.append(("Open", parent_id))
            else:
                actions.append(("Navigate", object_id))
            actions.append(("PickUp", object_id))
            for action_type, target in actions:
                step = gen.execute_and_record(
                    action_type=action_type,
                    target=target,
                    nl_description=f"{action_type} {target}",
                )
                if not step.feedback.success:
                    _RETRIEVAL_PLAN_CACHE[key] = False
                    return False
            _RETRIEVAL_PLAN_CACHE[key] = True
            return True
        finally:
            gen.close()


def _passive_probe_actions(
    *,
    target_parent: str | None,
    target_parent_openable: bool,
    target_object: str,
    navigate_object_after_open: bool = False,
) -> list[ExpectedAction]:
    """Build the probe plan while preserving the released default template."""
    actions: list[ExpectedAction] = []
    if target_parent:
        actions.append(ExpectedAction(action_type="Navigate", target=target_parent))
        if target_parent_openable:
            actions.append(ExpectedAction(action_type="Open", target=target_parent))
    if not target_parent_openable or navigate_object_after_open:
        actions.append(ExpectedAction(action_type="Navigate", target=target_object))
    actions.append(ExpectedAction(action_type="PickUp", target=target_object))
    return actions


def generate_l2_passive_retrieval(
    output_dir: str,
    context_scene: str = "FloorPlan1",
    noise_scene: str = "FloorPlan201",
    n_context_tasks: int = 3,
    n_noise_tasks: int = 5,
    seed: int = 42,
    target_parent_types: set[str] | None = None,
    target_object_types: set[str] | None = None,
    target_parent_id: str | None = None,
    target_object_id: str | None = None,
    episode_variant: str | None = None,
    suppress_zero_task_noise: bool = False,
    navigate_object_after_open: bool = False,
) -> Episode:
    """
    L2 类型1: 被动观察检索

    结构:
      Context Session (厨房): 执行 L1 任务，视野扫过特定物体
      Noise Session (客厅): 无关 L1 任务干扰
      Probe: "去厨房把那个 [物体] 拿过来"

    测试目标: 模型能否记住 Context 中被动观察到的物体位置
    """
    ep_id = f"ep_l2_passive_retrieval_{context_scene}_{noise_scene}"
    if episode_variant:
        variant = re.sub(r"[^A-Za-z0-9_-]+", "_", episode_variant).strip("_")
        if not variant:
            raise ValueError("episode_variant must contain an alphanumeric character")
        ep_id += f"_{variant}"
    ep_dir = os.path.join(output_dir, ep_id)

    builder = EpisodeBuilder(
        ep_id,
        "L2 Passive Observation Retrieval",
        "Test whether the model can remember an object location passively observed during the context session after distractor activity.",
        DifficultyLevel.L2,
    )
    if navigate_object_after_open:
        builder.set_scene_metadata(
            {
                "ood_template_variant": "passive_navigate_object_after_open",
            }
        )

    # ─── Context Session ────────────────────────
    ctx_dir = os.path.join(ep_dir, "context")
    print(f"\n  [Context] 启动 {context_scene} ...")
    gen_ctx = TrajectoryGenerator(context_scene, ctx_dir)

    if suppress_zero_task_noise and n_context_tasks == 0:
        ctx_steps, ctx_descs = [], []
    else:
        ctx_steps, ctx_descs = gen_ctx.generate_l1_noise(
            n_tasks=n_context_tasks, seed=seed
        )
    print(f"  [Context] 完成 {len(ctx_descs)} 个 L1 任务, {len(ctx_steps)} 步")

    # 找被动观察目标：视野中出现过但没被操作的 pickupable 物体
    operated = {
        s.action.target
        for s in ctx_steps
        if s.action.target and s.action.action_type in ("PickUp", "PutObject")
    }

    memory_target = None
    memory_step = None
    skip_types = {"Floor", "Wall", "Ceiling", "Window", "LightSwitch"}
    meta = gen_ctx.get_metadata()
    initial_meta = load_scene_metadata(context_scene)
    initial_parent_by_id = {
        obj["objectId"]: _metadata_parent(initial_meta, obj["objectId"])
        for obj in initial_meta["objects"]
    }
    pickupable_ids = {o["objectId"] for o in meta["objects"] if o.get("pickupable")}

    if target_parent_types is not None:
        candidate = _targeted_observation_candidate(
            meta,
            operated_ids=operated,
            parent_types=target_parent_types,
            object_types=target_object_types,
            target_parent_id=target_parent_id,
            target_object_id=target_object_id,
        )
        if candidate is None:
            print("  [WARN] 未找到指定父容器中的被动观察目标")
            gen_ctx.close()
            return None
        target_obj, parent_obj = candidate
        if not is_procthor_scene(context_scene) and not _retrieval_plan_executable(
            context_scene,
            target_obj["objectId"],
            parent_obj["objectId"],
            True,
        ):
            print("  [WARN] 指定被动观察目标没有可执行检索路径")
            gen_ctx.close()
            return None
        targeted_steps = _record_targeted_observation(gen_ctx, target_obj, parent_obj)
        if not targeted_steps:
            print("  [WARN] 指定父容器观察动作失败")
            gen_ctx.close()
            return None
        ctx_steps.extend(targeted_steps)
        meta = gen_ctx.get_metadata()
        pickupable_ids = {
            obj["objectId"] for obj in meta["objects"] if obj.get("pickupable")
        }
        for step in targeted_steps:
            for visible in step.visible_objects:
                if visible.object_id == target_obj["objectId"]:
                    memory_target = visible
                    memory_step = step.step_id
                    break
            if memory_target:
                break

    if memory_target is None:
        for s in reversed(ctx_steps):
            for vo in s.visible_objects:
                current_parent = _metadata_parent(meta, vo.object_id)
                initial_parent = initial_parent_by_id.get(vo.object_id)
                parent_obj = next(
                    (
                        obj
                        for obj in meta["objects"]
                        if obj["objectId"] == initial_parent
                    ),
                    None,
                )
                parent_openable = bool(parent_obj and parent_obj.get("openable"))
                if (
                    vo.object_id not in operated
                    and vo.object_type not in skip_types
                    and vo.object_id in pickupable_ids
                    and vo.distance < 3.0
                    and _has_clear_retrieval_parent(initial_parent)
                    and current_parent == initial_parent
                    and (
                        not is_procthor_scene(context_scene)
                        or _retrieval_plan_executable(
                            context_scene,
                            vo.object_id,
                            initial_parent,
                            parent_openable,
                        )
                    )
                ):
                    memory_target = vo
                    memory_step = s.step_id
                    break
            if memory_target:
                break

    if not memory_target:
        print("  [WARN] 未找到合适的被动观察目标")
        gen_ctx.close()
        return None

    # 查物体当前容器
    target_parent = None
    target_parent_openable = False
    target_parent = _metadata_parent(meta, memory_target.object_id)
    if target_parent:
        parent_obj = next(
            (obj for obj in meta["objects"] if obj["objectId"] == target_parent), None
        )
        target_parent_openable = bool(parent_obj and parent_obj.get("openable"))
    parent_name = (target_parent or "").split("|")[0]

    print(
        f"  [Memory Cue] {memory_target.object_type} (Step {memory_step} 被看到, 在 {parent_name})"
    )

    builder.add_memory_cue(
        MemoryCue(
            cue_id="cue_passive_obs",
            cue_type=MemoryCueType.PASSIVE_OBSERVATION,
            description=f"{memory_target.object_type} was passively observed during the context session",
            planted_in_session=1,
            planted_at_step=memory_step,
            object_id=memory_target.object_id,
            initial_location=target_parent,
            tested_in_probe="probe_retrieve",
        )
    )

    # 标记线索曝光步
    for s in ctx_steps:
        for vo in s.visible_objects:
            if vo.object_id == memory_target.object_id:
                s.memory_cue_exposed = MemoryCueExposure(
                    cue_id="cue_passive_obs",
                    exposure_type="peripheral_vision",
                    description=f"{memory_target.object_type} was visible on {parent_name}",
                )
                break

    leave_step = gen_ctx.generate_leave_room_step(get_room_type(noise_scene))
    ctx_steps.append(leave_step)
    gen_ctx.close()

    builder.add_session(
        session_name=f"{get_room_type(context_scene)} context task execution",
        scene=context_scene,
        trajectory_description=(
            f"Executed context tasks. During the session, {memory_target.object_type} "
            f"was passively observed on {parent_name}."
        ),
        steps=ctx_steps,
    )

    # ─── Noise Session ──────────────────────────
    noise_dir = os.path.join(ep_dir, "noise")
    print(f"\n  [Noise] 启动 {noise_scene} ...")
    gen_noise = TrajectoryGenerator(noise_scene, noise_dir)

    enter_step = gen_noise.generate_enter_room_step(get_room_type(context_scene))
    if suppress_zero_task_noise and n_noise_tasks == 0:
        noise_steps, noise_descs = [], []
    else:
        noise_steps, noise_descs = gen_noise.generate_l1_noise(
            n_tasks=n_noise_tasks, seed=seed + 100
        )
    noise_steps = [enter_step] + noise_steps
    print(f"  [Noise] 完成 {len(noise_descs)} 个任务, {len(noise_steps)} 步")

    expected_actions = _passive_probe_actions(
        target_parent=target_parent,
        target_parent_openable=target_parent_openable,
        target_object=memory_target.object_id,
        navigate_object_after_open=navigate_object_after_open,
    )

    probe = MicroProbe(
        probe_id="probe_retrieve",
        probe_level=DifficultyLevel.L2,
        instruction=f"Go to the {get_room_type(context_scene)} and bring back the {memory_target.object_type}.",
        context=(
            f"The agent passively observed {memory_target.object_type} on {parent_name} "
            f"during the context session and must retrieve its location after "
            f"{len(noise_steps)} distractor steps."
        ),
        scene=context_scene,
        expected_actions=expected_actions,
        optimal_steps=len(expected_actions),
        max_steps=20,
        evaluation_hooks=EvaluationHooks(
            success_condition=f"{memory_target.object_id} picked up",
            rar_triggers=[
                EvaluationTrigger(
                    trigger_condition=f"Navigate to locations other than {parent_name}",
                    penalty_type=PenaltyType.BLIND_NAVIGATION,
                    weight=1.0,
                    description="The agent should go directly to the remembered object location instead of searching blindly.",
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
