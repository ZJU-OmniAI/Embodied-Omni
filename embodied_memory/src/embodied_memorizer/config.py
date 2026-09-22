"""Portable defaults; dense embeddings can be selected explicitly."""

DEFAULT_CONFIG = {
    "embedding_model": "lexical-hash",
    "embedding_dim": 384,
    "spatial": {"max_nodes": 500},
    "scene": {"max_scenes": 200, "scene_similarity_threshold": 0.995},
    "event": {"max_events": 1000},
    "experience": {
        "max_experiences": 1000,
        "max_location_observations": 1000,
        "pending_window": 8,
        "allow_cross_namespace_fallback": False,
    },
    "retrieval": {"default_top_k": 3, "keyword_weight": 0.6, "embedding_weight": 0.4},
    "prompt": {
        "max_memory_tokens": 800,
        "auto_spatial_update": False,
        "auto_event_record": False,
        "auto_scene_record": False,
        "spatial_top_k": 10,
        "event_top_k": 5,
        "experience_top_k": 5,
        "scene_top_k": 2,
    },
    "embedding_device": "cpu",
}


def merge_config(user_config: dict, default: dict = None) -> dict:
    """Deep merge user config into default config."""
    if default is None:
        default = DEFAULT_CONFIG
    result = default.copy()
    for key, value in user_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(value, result[key])
        else:
            result[key] = value
    return result
