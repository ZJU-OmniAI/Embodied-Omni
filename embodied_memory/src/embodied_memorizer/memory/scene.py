"""
情景记忆：存储场景快照（caption + 图像路径 + 可见物体）。

检索分两级（均在本模块内完成）：
  Level 1  _keyword_search()    在 caption 和 visible_objects 中匹配关键词
  Level 2  _embedding_search()  caption embedding 语义搜索，关键词不足时补齐

EmbeddingEngine 由 MemorySystem 注入，与其他模块共享同一实例。
"""

import uuid
import numpy as np
from typing import Dict, List, Optional
from PIL import Image, ImageOps


class SceneMemory:
    """
    场景快照存储。
    """

    def __init__(self, config: dict = None, embedding_engine=None):
        config = config or {}
        self.max_scenes = config.get("max_scenes", 200)
        self.scene_similarity_threshold = config.get(
            "scene_similarity_threshold", 0.995
        )
        if not 0 < self.scene_similarity_threshold <= 1:
            raise ValueError("scene_similarity_threshold must be in (0, 1]")
        self.scenes: Dict[str, dict] = {}
        self._emb = embedding_engine  # 共享 EmbeddingEngine，可为 None
        self._emb_cache: dict = {}  # scene_id -> np.ndarray
        self._visual_cache: dict = {}

    def reset(self):
        self.scenes.clear()
        self._emb_cache.clear()
        self._visual_cache.clear()

    @staticmethod
    def _visual_feature(image_path):
        """Normalized RGB thumbnail; no metadata, captions or model downloads.

        Unreadable/missing RGB fails open: keep the observation rather than
        infer visual similarity from its filename or caption alone.
        """
        if not image_path:
            return None
        try:
            with Image.open(image_path) as image:
                pixels = np.asarray(
                    ImageOps.exif_transpose(image).convert("RGB").resize((32, 32)),
                    dtype=np.float32,
                ).reshape(-1)
        except (OSError, ValueError):
            return None
        # The offset makes a completely black observation well-defined.
        pixels += 1.0
        return pixels / np.linalg.norm(pixels)

    def add_scene(
        self,
        caption: str,
        step: int,
        image_path: str = None,
        visible_objects: List[str] = None,
        scene_id: str = None,
        scope: str = None,
    ) -> str:
        """添加场景快照。"""
        feature = self._visual_feature(image_path)
        entities = {name.strip().casefold() for name in visible_objects or []}
        normalized_caption = " ".join(caption.casefold().split())
        if feature is not None and scene_id is None:
            for sid, scene in reversed(list(self.scenes.items())):
                if scene.get("scope") != scope:
                    continue
                # Conservatively retain new entities and changed instance/state
                # descriptions, even when the overall RGB is almost identical.
                known = {name.strip().casefold() for name in scene["visible_objects"]}
                if not entities.issubset(known) or normalized_caption != " ".join(
                    scene["caption"].casefold().split()
                ):
                    continue
                previous = self._visual_cache.get(sid)
                if (
                    previous is not None
                    and float(np.dot(feature, previous))
                    >= self.scene_similarity_threshold
                ):
                    return sid
        sid = scene_id or f"scene_{uuid.uuid4().hex[:8]}"
        self.scenes[sid] = {
            "step": step,
            "caption": caption,
            "image_path": image_path,
            "visible_objects": visible_objects or [],
            "scope": scope,
        }
        self._emb_cache.pop(sid, None)
        self._visual_cache.pop(sid, None)
        if feature is not None:
            self._visual_cache[sid] = feature

        if len(self.scenes) > self.max_scenes:
            oldest_id = min(self.scenes, key=lambda k: self.scenes[k]["step"])
            self.remove_scene(oldest_id)

        return sid

    def remove_scene(self, scene_id: str) -> bool:
        if scene_id not in self.scenes:
            return False
        del self.scenes[scene_id]
        self._emb_cache.pop(scene_id, None)
        self._visual_cache.pop(scene_id, None)
        return True

    # ── 两级检索 ──

    def _keyword_search(self, query: str, top_k: int, scope: str = None) -> List[dict]:
        """Level 1：在 caption 和 visible_objects 中匹配关键词，最新优先。"""
        kw = query.lower()
        results = []
        for sid, scene in sorted(
            self.scenes.items(), key=lambda x: x[1]["step"], reverse=True
        ):
            if scope is not None and scene.get("scope") != scope:
                continue
            caption_match = kw in scene.get("caption", "").lower()
            obj_match = any(
                kw in obj.lower() for obj in scene.get("visible_objects", [])
            )
            if caption_match or obj_match:
                results.append({"id": sid, **scene})
                if len(results) >= top_k:
                    break
        return results

    def _embedding_search(
        self, query: str, top_k: int, exclude: set, scope: str = None
    ) -> List[dict]:
        """Level 2：caption embedding 余弦相似度搜索，跳过 exclude 中的场景。"""
        if self._emb is None or not self.scenes:
            return []

        # 增量计算缺失的 embedding（仅对有实质 caption 的场景）
        for sid, scene in self.scenes.items():
            if scope is not None and scene.get("scope") != scope:
                continue
            if sid not in self._emb_cache:
                text = scene.get("caption", "")
                if text:
                    self._emb_cache[sid] = self._emb.encode_single(text)

        candidates = [
            (sid, self._emb_cache[sid])
            for sid in self.scenes
            if sid in self._emb_cache
            and sid not in exclude
            and (scope is None or self.scenes[sid].get("scope") == scope)
        ]
        if not candidates:
            return []

        query_emb = self._emb.encode_single(query)
        emb_matrix = np.array([c[1] for c in candidates])
        scores = self._emb.batch_cosine_similarity(query_emb, emb_matrix)

        results = []
        for idx in np.argsort(scores)[::-1]:
            if scores[idx] > 0.1:
                sid = candidates[idx][0]
                results.append({"id": sid, **self.scenes[sid]})
                if len(results) >= top_k:
                    break
        return results

    def query(self, query: str, top_k: int = 3, scope: str = None) -> List[dict]:
        """两级检索：关键词优先，不足时 embedding 补齐。"""
        if top_k <= 0:
            return []
        results = self._keyword_search(query, top_k, scope)
        if len(results) < top_k:
            exclude = {r["id"] for r in results}
            results += self._embedding_search(
                query, top_k - len(results), exclude, scope
            )
        return results[:top_k]

    def query_recent(self, top_k: int = 3, scope: str = None) -> List[dict]:
        """按最新优先返回 top_k 条，供无 query 时的兜底使用。"""
        sorted_scenes = sorted(
            (
                (sid, scene)
                for sid, scene in self.scenes.items()
                if scope is None or scene.get("scope") == scope
            ),
            key=lambda x: x[1]["step"],
            reverse=True,
        )
        return [{"id": sid, **data} for sid, data in sorted_scenes[: max(0, top_k)]]

    # ── Prompt 生成 ──

    def to_text(self, query: str = "", top_k: int = 3) -> str:
        """
        生成注入 prompt 的场景文本。
        query 非空时走两级检索；为空时退化为取最新 top_k 条。
        """
        if not self.scenes:
            return "No scene memory recorded yet."

        results = self.query(query, top_k) if query else self.query_recent(top_k)

        lines = ["[Scene Memory]"]
        for scene in results:
            lines.append(f"- Step {scene['step']}: {scene['caption']}")
            objs = ", ".join(scene.get("visible_objects", [])[:8])
            if objs:
                lines.append(f"  Objects: {objs}")
        return "\n".join(lines)
