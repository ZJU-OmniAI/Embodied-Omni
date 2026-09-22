"""L2 类型3: 动态位置追踪"""

import json
import os
import re
from pathlib import Path

from ...scene_utils import TrajectoryGenerator, get_room_type, load_scene_metadata
from ...schema import (
    Episode,
    MicroProbe,
    MemoryCue,
    ExpectedAction,
    EvaluationHooks,
    EvaluationTrigger,
    DifficultyLevel,
    MemoryCueType,
    PenaltyType,
    StateChange,
)
from ..builder import EpisodeBuilder


_PUT_COMPAT_CACHE: dict[str, list[str]] | None = None


def _load_put_compat() -> dict[str, list[str]]:
    global _PUT_COMPAT_CACHE
    if _PUT_COMPAT_CACHE is None:
        compat_path = (
            Path(__file__).resolve().parents[2] / "assets" / "put_compat_ai2thor.json"
        )
        with open(compat_path, "r", encoding="utf-8") as f:
            _PUT_COMPAT_CACHE = json.load(f)
    return _PUT_COMPAT_CACHE


def _parent_map(meta: dict) -> dict[str, str]:
    parents = {}
    for obj in meta["objects"]:
        parent_ids = obj.get("parentReceptacles") or []
        if parent_ids:
            parents[obj["objectId"]] = parent_ids[0]
    return parents


def _object_blocked_by_closed_parent(obj: dict, by_id: dict[str, dict]) -> bool:
    for parent_id in obj.get("parentReceptacles") or []:
        parent = by_id.get(parent_id)
        if parent and parent.get("openable") and not parent.get("isOpen"):
            return True
    return False


def _location_type(location_id: str | None, meta: dict) -> str:
    if not location_id:
        return ""
    obj = next((o for o in meta["objects"] if o["objectId"] == location_id), None)
    return (obj or {}).get("objectType") or location_id.split("|")[0]


def _is_meaningful_location_change(
    old_id: str | None, new_id: str | None, meta: dict
) -> bool:
    if not old_id or not new_id or old_id == new_id:
        return False
    return _location_type(old_id, meta) != _location_type(new_id, meta)


def _stable_receptacles(meta: dict, *, allow_openable: bool = False) -> list[dict]:
    priority = {
        "CounterTop": 0,
        "DiningTable": 1,
        "CoffeeTable": 2,
        "SideTable": 3,
        "Desk": 4,
        "Shelf": 5,
        "TVStand": 6,
        "Dresser": 7,
        "Bed": 8,
        "Sofa": 9,
        "ArmChair": 10,
        "Ottoman": 11,
        "StoveBurner": 12,
        "SinkBasin": 13,
    }
    return sorted(
        [
            obj
            for obj in meta["objects"]
            if obj.get("receptacle")
            and (not obj.get("openable") or (allow_openable and not obj.get("isOpen")))
            and not obj.get("pickupable")
            and obj.get("objectType") not in {"Floor", "GarbageCan"}
        ],
        key=lambda obj: (
            priority.get(obj.get("objectType"), 100),
            not bool(obj.get("visible")),
            obj.get("objectId", ""),
        ),
    )


def _append_directed_tracking_move(
    gen_ctx: TrajectoryGenerator,
    ctx_steps: list,
    ctx_descs: list[str],
    *,
    target_object_id: str | None = None,
    target_destination_id: str | None = None,
    allow_openable_destination: bool = False,
) -> tuple[dict, str, str] | None:
    """Append one executed pick-and-place move when random L1 produced none."""
    compat = _load_put_compat()
    meta = gen_ctx.get_metadata()
    by_id = {obj["objectId"]: obj for obj in meta["objects"]}
    parent_by_id = _parent_map(meta)

    objects = sorted(
        [
            obj
            for obj in meta["objects"]
            if obj.get("pickupable")
            and not obj.get("isPickedUp")
            and obj.get("objectType") in compat
            and not _object_blocked_by_closed_parent(obj, by_id)
        ],
        key=lambda obj: (
            not bool(obj.get("visible")),
            obj.get("objectType", ""),
            obj.get("objectId", ""),
        ),
    )
    receptacles = _stable_receptacles(
        meta,
        allow_openable=allow_openable_destination,
    )

    for obj in objects:
        if target_object_id and obj["objectId"] != target_object_id:
            continue
        old_parent = gen_ctx._find_object_current_parent(
            obj["objectId"]
        ) or parent_by_id.get(obj["objectId"], "")
        if not old_parent:
            continue
        compatible_targets = set(compat.get(obj["objectType"], []))
        for receptacle in receptacles:
            if (
                target_destination_id
                and receptacle["objectId"] != target_destination_id
            ):
                continue
            if not _is_meaningful_location_change(
                old_parent, receptacle["objectId"], meta
            ):
                continue
            if receptacle["objectType"] not in compatible_targets:
                continue

            counter_before = gen_ctx.step_counter
            actions = [
                {
                    "action": "Navigate",
                    "target": obj["objectId"],
                    "nl": f"Navigate to {obj['objectType']}",
                },
                {
                    "action": "PickUp",
                    "target": obj["objectId"],
                    "nl": f"Pick up {obj['objectType']}",
                },
                {
                    "action": "Navigate",
                    "target": receptacle["objectId"],
                    "nl": f"Navigate to {receptacle['objectType']}",
                },
            ]
            if receptacle.get("openable") and not receptacle.get("isOpen"):
                actions.append(
                    {
                        "action": "Open",
                        "target": receptacle["objectId"],
                        "nl": f"Open {receptacle['objectType']}",
                    }
                )
            actions.append(
                {
                    "action": "PutObject",
                    "target": receptacle["objectId"],
                    "nl": f"Move {obj['objectType']} to {receptacle['objectType']}",
                }
            )
            if receptacle.get("openable"):
                actions.append(
                    {
                        "action": "Close",
                        "target": receptacle["objectId"],
                        "nl": f"Close {receptacle['objectType']}",
                    }
                )
            sub_steps = gen_ctx.generate_task_trajectory(actions)
            if len(sub_steps) == len(actions) and all(
                step.feedback.success for step in sub_steps
            ):
                relocated = (
                    gen_ctx._find_object_current_parent(obj["objectId"])
                    == receptacle["objectId"]
                    if receptacle.get("openable")
                    else gen_ctx._relocated_object_is_pickupable(
                        obj["objectId"], receptacle["objectId"]
                    )
                )
                if not relocated:
                    gen_ctx._discard_steps(sub_steps, counter_before)
                    return None
                ctx_steps.extend(sub_steps)
                ctx_descs.append(
                    f"[tracking_move] Move {obj['objectType']} to {receptacle['objectType']}"
                )
                moved_obj = gen_ctx._get_obj_meta(obj["objectId"]) or obj
                return moved_obj, old_parent, receptacle["objectId"]

            gen_ctx._discard_steps(sub_steps, counter_before)
            gen_ctx.ensure_hand_empty()

    return None


def generate_l2_dynamic_tracking(
    output_dir: str,
    context_scene: str = "FloorPlan1",
    noise_scene: str = "FloorPlan201",
    n_context_tasks: int = 3,
    n_noise_tasks: int = 5,
    seed: int = 42,
    target_object_id: str | None = None,
    target_destination_id: str | None = None,
    episode_variant: str | None = None,
    suppress_zero_task_noise: bool = False,
    allow_openable_destination: bool = False,
) -> Episode:
    """
    L2 类型3: 动态位置追踪

    结构:
      Context Session: Agent 执行 L1 任务，搬动物体 A 从位置 X 到位置 Y
      Noise Session: 无关 L1 任务干扰
      Probe: "去拿物体 A" → 应去位置 Y（当前位置），非 X（初始位置）

    测试目标: 模型能否追踪自己操作引起的物体位移
    """
    ep_id = f"ep_l2_dynamic_tracking_{context_scene}_{noise_scene}"
    if episode_variant:
        variant = re.sub(r"[^A-Za-z0-9_-]+", "_", episode_variant).strip("_")
        if not variant:
            raise ValueError("episode_variant must contain an alphanumeric character")
        ep_id += f"_{variant}"
    ep_dir = os.path.join(output_dir, ep_id)

    builder = EpisodeBuilder(
        ep_id,
        "L2 Dynamic Location Tracking",
        "Test whether the model can track an object it previously moved and remember its current location after distractor activity.",
        DifficultyLevel.L2,
    )
    if allow_openable_destination:
        builder.set_scene_metadata(
            {
                "ood_template_variant": "dynamic_openable_destination",
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

    exact_move = None
    if target_object_id or target_destination_id:
        if not (target_object_id and target_destination_id):
            gen_ctx.close()
            raise ValueError(
                "target_object_id and target_destination_id must be provided together"
            )
        exact_move = _append_directed_tracking_move(
            gen_ctx,
            ctx_steps,
            ctx_descs,
            target_object_id=target_object_id,
            target_destination_id=target_destination_id,
            allow_openable_destination=allow_openable_destination,
        )
        if exact_move is None:
            print("  [WARN] 指定动态追踪移动不可执行")
            gen_ctx.close()
            return None

    # 找被搬动的物体（PickUp 后跟 PutObject 的物体）
    moved = {}
    known_parent = _parent_map(load_scene_metadata(context_scene))
    held_object_id = None
    held_old_parent = None
    for s in ctx_steps:
        if s.action.action_type == "PickUp" and s.action.target:
            held_object_id = s.action.target
            held_old_parent = known_parent.get(held_object_id)
        elif s.action.action_type == "PutObject" and s.action.target and held_object_id:
            moved[held_object_id] = {"from": held_old_parent, "to": s.action.target}
            known_parent[held_object_id] = s.action.target
            held_object_id = None
            held_old_parent = None

    tracking_target = None
    tracking_old_loc = None
    tracking_new_loc = None
    if exact_move:
        tracking_target, tracking_old_loc, tracking_new_loc = exact_move
    meta = gen_ctx.get_metadata()
    for oid, info in moved.items():
        if not _is_meaningful_location_change(info.get("from"), info.get("to"), meta):
            continue
        target_receptacle = next(
            (obj for obj in meta["objects"] if obj["objectId"] == info["to"]),
            None,
        )
        if (
            not target_receptacle
            or not target_receptacle.get("receptacle")
            or (target_receptacle.get("openable") and not allow_openable_destination)
            or target_receptacle.get("pickupable")
            or target_receptacle.get("objectType") in {"Floor", "GarbageCan"}
        ):
            continue
        for obj in meta["objects"]:
            if obj["objectId"] == oid and obj.get("pickupable"):
                relocated = (
                    gen_ctx._find_object_current_parent(oid) == info["to"]
                    if target_receptacle.get("openable")
                    else gen_ctx._relocated_object_is_pickupable(oid, info["to"])
                )
                if not relocated:
                    continue
                tracking_target = obj
                tracking_old_loc = info.get("from")
                tracking_new_loc = info["to"]
                break
        if tracking_target:
            break

    if not tracking_target:
        directed_move = _append_directed_tracking_move(
            gen_ctx,
            ctx_steps,
            ctx_descs,
            allow_openable_destination=allow_openable_destination,
        )
        if directed_move:
            tracking_target, tracking_old_loc, tracking_new_loc = directed_move
        else:
            print("  [WARN] 未找到被搬动的物体")
            gen_ctx.close()
            return None

    initial_parent = tracking_old_loc
    init_name = _location_type(initial_parent, meta)
    new_name = _location_type(tracking_new_loc, meta)
    print(f"  [Tracking] {tracking_target['objectType']}: {init_name} → {new_name}")

    builder.add_memory_cue(
        MemoryCue(
            cue_id="cue_moved_object",
            cue_type=MemoryCueType.SELF_ACTION,
            description=f"Agent moved {tracking_target['objectType']} from {init_name} to {new_name}",
            planted_in_session=1,
            planted_at_step=-1,
            object_id=tracking_target["objectId"],
            initial_location=initial_parent,
            tested_in_probe="probe_track_location",
        )
    )

    leave_step = gen_ctx.generate_leave_room_step(get_room_type(noise_scene))
    ctx_steps.append(leave_step)
    gen_ctx.close()

    builder.add_session(
        session_name=f"{get_room_type(context_scene)} context task execution",
        scene=context_scene,
        trajectory_description=(
            f"Executed context tasks and moved {tracking_target['objectType']} "
            f"from {init_name} to {new_name}."
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
            n_tasks=n_noise_tasks, seed=seed + 300
        )
    noise_steps = [enter_step] + noise_steps

    destination_openable = bool(
        next(
            (
                obj.get("openable")
                for obj in meta.get("objects") or []
                if obj.get("objectId") == tracking_new_loc
            ),
            False,
        )
    )
    expected_actions = [
        ExpectedAction(action_type="Navigate", target=tracking_new_loc),
    ]
    if destination_openable:
        expected_actions.append(
            ExpectedAction(action_type="Open", target=tracking_new_loc)
        )
    expected_actions.extend(
        [
            ExpectedAction(action_type="Navigate", target=tracking_target["objectId"]),
            ExpectedAction(action_type="PickUp", target=tracking_target["objectId"]),
        ]
    )

    probe = MicroProbe(
        probe_id="probe_track_location",
        probe_level=DifficultyLevel.L2,
        instruction=f"Go to the {get_room_type(context_scene)} and bring back the {tracking_target['objectType']}.",
        context=(
            f"The agent moved {tracking_target['objectType']} from {init_name} "
            f"to {new_name} during the context session and must track its current location."
        ),
        scene=context_scene,
        expected_actions=expected_actions,
        optimal_steps=len(expected_actions),
        max_steps=20,
        evaluation_hooks=EvaluationHooks(
            success_condition=f"{tracking_target['objectId']} picked up",
            rar_triggers=[
                EvaluationTrigger(
                    trigger_condition=f"Navigate to {init_name} (stale initial location)",
                    penalty_type=PenaltyType.STALE_SPATIAL_MEMORY,
                    weight=3.0,
                    description=(
                        f"Went to stale initial location {init_name} instead of "
                        f"current location {new_name}; failed to track the moved object."
                    ),
                ),
            ],
        ),
        state_changes=[
            StateChange(
                object_id=tracking_target["objectId"],
                change_type="position_change",
                old_value=initial_parent or "",
                new_value=tracking_new_loc,
                description=f"{tracking_target['objectType']} was moved from {init_name} to {new_name} during Context.",
            ),
        ],
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
