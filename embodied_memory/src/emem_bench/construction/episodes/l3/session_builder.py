"""
构建 L3 Context Session 和 Noise Session。

  _build_constraint_failure_session: 构建单个"约束失败"Session
  _build_noise_sessions:             构建多个无关噪声 Session
"""

import os
from typing import Optional

import numpy as np

from ...scene_utils import TrajectoryGenerator, get_room_type
from ...experience_templates import ConstraintRule
from .rule_helpers import is_owner_habit_rule
from .action_sequences import (
    materialize_failure_action_sequence,
    materialize_correction_actions,
)
from .object_finders import find_safe_container, find_all_safe_containers

_ALL_SCENES = (
    [f"FloorPlan{i}" for i in range(1, 31)]
    + [f"FloorPlan{200 + i}" for i in range(1, 31)]
    + [f"FloorPlan{300 + i}" for i in range(1, 31)]
    + [f"FloorPlan{400 + i}" for i in range(1, 31)]
)


def build_constraint_failure_session(
    gen: TrajectoryGenerator,
    session_idx: int,
    rule: ConstraintRule,
    failure_object: dict,
    target_object: dict,
    safe_object: Optional[dict],
    preferred_object: Optional[dict],
    scene: str,
    seed: int,
) -> tuple[list, str]:
    """
    构建一个"约束失败"Context Session。

    结构：
      Part A: 背景 L1 任务
      Part B: 违反约束的动作 → 失败反馈
      Part C: 恢复 L1 任务

    返回: (steps, trajectory_description)
    """
    steps = []
    descriptions = []

    # ─── Part A: 背景 L1 任务 ──────────────────────────────
    ctx_steps, ctx_descs = gen.generate_l1_noise(n_tasks=1, seed=seed)
    steps.extend(ctx_steps)
    descriptions.extend(ctx_descs)

    # ─── Part B: 插入失败事件 ──────────────────────────────
    if is_owner_habit_rule(rule):
        failure_desc = (
            f"tried placing {failure_object['objectType']} on {target_object['objectType']}, "
            "but that placement did not match the household habit"
        )
    else:
        failure_desc = f"tried interacting {failure_object['objectType']} with {target_object['objectType']}"

    failure_actions = materialize_failure_action_sequence(
        rule=rule,
        failure_object=failure_object,
        target_object=target_object,
        session_idx=session_idx,
        safe_object=safe_object,
        preferred_object=preferred_object,
    )
    steps.extend(gen.generate_task_trajectory(failure_actions))
    descriptions.append(failure_desc)

    # ─── Part B2: 矫正（模拟人类干预，展示正确行为）─────────────
    if is_owner_habit_rule(rule):
        correction_actions = materialize_correction_actions(
            rule=rule,
            failure_object=failure_object,
            safe_object=safe_object,
            preferred_object=preferred_object,
        )
        steps.extend(gen.generate_task_trajectory(correction_actions))
    else:
        # physical_constraint: 先记录人工干预步骤，再用自动重试确保放置成功
        steps.extend(
            gen.generate_task_trajectory(
                [
                    {
                        "action": "HumanIntervention",
                        "nl": f"Human intervention: task rejected - {rule.incompatibility_rule}",
                    }
                ]
            )
        )
        # 用实时场景状态获取全量备选容器，依次重试直到放置成功
        live_meta = gen.get_metadata()
        safe_candidates = find_all_safe_containers(
            live_meta,
            failure_object_type=failure_object["objectType"],
            avoid_object_id=failure_object["objectId"],
        )
        steps.extend(
            gen.put_object_with_fallback(
                receptacle_ids=[o["objectId"] for o in safe_candidates],
                obj_nl=failure_object["objectType"],
            )
        )

    # ─── Part C: 恢复 L1 任务 ───────────────────────────────
    recovery_steps, recovery_descs = gen.generate_l1_noise(n_tasks=1, seed=seed + 50)
    steps.extend(recovery_steps)
    descriptions.extend(recovery_descs)

    trajectory_desc = f"Executed L1 tasks; during the session, {failure_desc}; the action failed and was corrected by human intervention."

    return steps, trajectory_desc


def build_noise_sessions(
    output_dir: str,
    n_sessions: int = 2,
    n_tasks_per_session: int = 3,
    seed: int = 1000,
    used_scenes: set[str] | None = None,
    scene_pool: list[str] | None = None,
) -> list[dict]:
    """构建多个噪声 Session，从全部场景中随机选取，排除已用场景。"""
    rng = np.random.RandomState(seed)
    source_pool = scene_pool or _ALL_SCENES
    pool = [s for s in source_pool if s not in (used_scenes or set())]
    chosen = rng.choice(pool, size=min(n_sessions, len(pool)), replace=False).tolist()

    noise_sessions = []
    for i, scene in enumerate(chosen):
        ep_dir = os.path.join(output_dir, f"ep_l3_noise_s{i + 1}")
        gen = TrajectoryGenerator(scene, ep_dir)
        steps, descs = gen.generate_l1_noise(
            n_tasks=n_tasks_per_session, seed=seed + i * 100
        )
        noise_sessions.append({"scene": scene, "steps": steps, "descriptions": descs})
        gen.close()

    return noise_sessions
