"""数据生成辅助工具"""

import json
import os

from .schema import Episode, save_episode as _save_episode_raw
from .scene_utils import get_room_type
from .tasks import Task


def save_episode_to_dir(episode: Episode, output_dir: str) -> str:
    """将 Episode 主 JSON 保存到 output_dir/{episode_id}/{episode_id}.json。

    生成器内部已在 ep_dir 下创建 session_*/obs_xxx.png 等文件，
    主 JSON 也应放在同一子目录内，保持目录结构一致。

    Returns:
        保存的 JSON 文件路径
    """
    ep_dir = os.path.join(output_dir, episode.episode_id)
    os.makedirs(ep_dir, exist_ok=True)
    path = os.path.join(ep_dir, f"{episode.episode_id}.json")
    _save_episode_raw(episode, path)
    return path


def save_tasks_episode(
    tasks: list[Task],
    scene: str,
    seed: int,
    output_dir: str,
    level: str,
) -> str:
    """将任务列表按 Episode JSON 格式保存到 output_dir 子目录。

    目录结构:
        output_dir/
          ep_{level.lower()}_{scene}_seed{seed}/
            ep_{level.lower()}_{scene}_seed{seed}.json

    Args:
        level: "L0" 或 "L1"（决定 episode_id 前缀和描述文字）

    Returns:
        保存的 JSON 文件路径
    """
    lvl = level.lower()
    episode_id = f"ep_{lvl}_{scene}_seed{seed}"
    ep_dir = os.path.join(output_dir, episode_id)
    os.makedirs(ep_dir, exist_ok=True)

    room_type = get_room_type(scene)

    sessions = [
        {
            "session_id": f"session_{i + 1}",
            "session_name": t.name,
            "scene": scene,
            "room_type": room_type,
            "level": t.level,
            "task_type": t.task_type,
            "context_trajectory": {
                "description": t.name,
                "total_steps": t.step_count,
                "steps": [
                    {
                        "step_id": j,
                        "action": {
                            "action_type": s.action,
                            "target": s.target,
                            "natural_language": s.nl,
                        },
                        "feedback": {"success": True, "message": ""},
                    }
                    for j, s in enumerate(t.steps)
                ],
            },
        }
        for i, t in enumerate(tasks)
    ]

    episode = {
        "episode_id": episode_id,
        "episode_name": f"{level}任务集 — {scene}",
        "description": f"场景 {scene} 的 {level} 任务采样（seed={seed}）",
        "difficulty": level,
        "hidden_rules": [],
        "memory_cues": [],
        "sessions": sessions,
        "macro_probe": None,
    }

    out_path = os.path.join(ep_dir, f"{episode_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(episode, f, ensure_ascii=False, indent=2)
    return out_path


def render_and_save_tasks(
    tasks: list[Task],
    scene: str,
    seed: int,
    output_dir: str,
    level: str,
) -> str:
    """在 AI2-THOR 模拟器中执行任务列表并渲染，保存完整 Episode（含图像）。

    每个 Task 对应一个 Session，图像保存在各自的 session_N/ 子目录。
    复用同一个模拟器实例（避免重复启动的开销），任务间调用 ensure_hand_empty()。

    Args:
        level: "L0" 或 "L1"

    Returns:
        保存的 JSON 文件路径
    """
    from .scene_utils.simulator import TrajectoryGenerator
    from .episodes.builder import EpisodeBuilder
    from .schema import DifficultyLevel

    lvl = level.lower()
    episode_id = f"ep_{lvl}_{scene}_seed{seed}"
    ep_dir = os.path.join(output_dir, episode_id)
    os.makedirs(ep_dir, exist_ok=True)

    difficulty = DifficultyLevel.L0 if level == "L0" else DifficultyLevel.L1
    builder = EpisodeBuilder(
        episode_id=episode_id,
        name=f"{level}任务集 — {scene}",
        description=f"场景 {scene} 的 {level} 任务渲染（seed={seed}）",
        difficulty=difficulty,
    )

    # 复用同一模拟器实例，切换 output_dir 来区分 session 图像
    gen = TrajectoryGenerator(scene, ep_dir)
    try:
        for i, task in enumerate(tasks):
            session_dir = os.path.join(ep_dir, f"session_{i + 1}")
            os.makedirs(session_dir, exist_ok=True)
            gen.output_dir = session_dir  # 图像写入 session 子目录
            gen.step_counter = 0  # 每个 session 从 step 0 开始

            action_dicts = [s.to_dict() for s in task.steps]
            traj_steps = gen.generate_task_trajectory(action_dicts)

            builder.add_session(
                session_name=task.name,
                scene=scene,
                trajectory_description=task.name,
                steps=traj_steps,
            )
            gen.ensure_hand_empty()
    finally:
        gen.close()

    episode = builder.build()
    return save_episode_to_dir(episode, output_dir)


# 向后兼容别名
def save_l1_episode(tasks: list[Task], scene: str, seed: int, output_dir: str) -> str:
    return save_tasks_episode(tasks, scene, seed, output_dir, level="L1")
