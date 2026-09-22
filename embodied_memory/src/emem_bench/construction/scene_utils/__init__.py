from .metadata import (
    METADATA_ROOT,
    ROOM_TYPE_MAP,
    get_room_type,
    load_scene_metadata,
    load_agent_origin,
    find_objects_by_type,
    find_objects_by_property,
)
from .procthor import (
    make_procthor_scene_ref,
    parse_procthor_scene_ref,
    resolve_controller_scene,
    list_procthor_scene_refs,
    list_procthor_scene_refs_all,
    procthor_split_size,
)
from .query import SceneQuery


# simulator 依赖 ai2thor，延迟导入避免没装 ai2thor 时整个包挂掉
def __getattr__(name):
    if name == "TrajectoryGenerator":
        from .simulator import TrajectoryGenerator

        return TrajectoryGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
