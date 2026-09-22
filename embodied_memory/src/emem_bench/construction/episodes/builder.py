"""Episode 构建器"""

from typing import Optional

from ..schema import (
    Episode,
    Session,
    ContextTrajectory,
    TrajectoryStep,
    MicroProbe,
    MacroProbe,
    HiddenRule,
    MemoryCue,
    DifficultyLevel,
)
from ..scene_utils import get_room_type


class EpisodeBuilder:
    """
    Episode 构建流式 API。

    用法：
        builder = EpisodeBuilder("ep_001", "测试Episode", "...", DifficultyLevel.L2)
        builder.add_memory_cue(...)
        builder.add_session(...)
        episode = builder.build()
    """

    def __init__(
        self, episode_id: str, name: str, description: str, difficulty: DifficultyLevel
    ):
        self.episode = Episode(
            episode_id=episode_id,
            episode_name=name,
            description=description,
            difficulty=difficulty,
        )
        self._current_session_idx = 0

    def add_hidden_rule(self, rule: HiddenRule):
        self.episode.hidden_rules.append(rule)
        return self

    def add_memory_cue(self, cue: MemoryCue):
        self.episode.memory_cues.append(cue)
        return self

    def add_session(
        self,
        session_name: str,
        scene: str,
        trajectory_description: str,
        steps: list[TrajectoryStep],
        micro_probe: Optional[MicroProbe] = None,
    ):
        self._current_session_idx += 1
        session = Session(
            session_id=f"session_{self._current_session_idx}",
            session_name=session_name,
            scene=scene,
            room_type=get_room_type(scene),
            context_trajectory=ContextTrajectory(
                description=trajectory_description,
                total_steps=len(steps),
                steps=steps,
            ),
            micro_probe=micro_probe,
        )
        self.episode.sessions.append(session)
        return self

    def set_macro_probe(self, macro_probe: MacroProbe):
        self.episode.macro_probe = macro_probe
        return self

    def set_scene_metadata(self, metadata: dict):
        self.episode.scene_metadata = metadata
        return self

    def build(self) -> Episode:
        return self.episode
