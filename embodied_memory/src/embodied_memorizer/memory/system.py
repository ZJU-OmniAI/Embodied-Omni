"""
统一记忆系统：管理三层记忆，共享 EmbeddingEngine，提供跨层引用与统一写入接口。
"""

from typing import List, Optional

from .spatial import SpatialMemory
from .scene import SceneMemory
from .event import EventMemory
from .experience import ExperienceMemory
from ..embedding_utils import EmbeddingEngine


class MemorySystem:
    """Unified memory interface with raw and consolidated memory layers."""

    def __init__(self, config: dict = None):
        config = config or {}
        # 共享 EmbeddingEngine，懒加载（首次检索时才加载模型）
        emb = EmbeddingEngine(
            model_name=config.get(
                "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            device=config.get("embedding_device", "cuda"),
            base_url=config.get("embedding_base_url", None),
            api_key=config.get("embedding_api_key", "dummy"),
        )
        self.spatial = SpatialMemory(config.get("spatial", {}), embedding_engine=emb)
        self.scene = SceneMemory(config.get("scene", {}), embedding_engine=emb)
        self.event = EventMemory(config.get("event", {}), embedding_engine=emb)
        self.experience = ExperienceMemory(config.get("experience", {}))
        self.current_step = 0
        self.current_image_path: Optional[str] = None  # 当前步骤的观测图像路径

    def reset(self):
        self.spatial.reset()
        self.scene.reset()
        self.event.reset()
        self.experience.reset()
        self.current_step = 0
        self.current_image_path = None

    def step(self):
        self.current_step += 1

    # ── 写入接口 ──

    def add_spatial(
        self,
        name: str,
        node_type: str = "object",
        relations: Optional[List[dict]] = None,
        properties: dict = None,
        scope: str = None,
    ) -> str:
        """写入空间记忆节点，并按模型声明的 relations 建边。

        relations 格式：[{"target": "CounterTop", "type": "on"}, ...]
        target 节点不存在时自动 upsert（node_type 默认 "object"）。
        """
        nid = self.spatial.add_node(
            name=name,
            node_type=node_type,
            properties=properties or {},
            step=self.current_step,
            scope=scope,
        )
        for rel in relations or []:
            target = rel.get("target", "").strip()
            rel_type = rel.get("type", "related").strip()
            if not target:
                continue
            tid = self.spatial.add_node(
                name=target,
                node_type="object",
                properties={},
                step=self.current_step,
                scope=scope,
            )
            self.spatial.add_edge(nid, tid, rel_type)
        return nid

    def update_spatial(
        self,
        name: str,
        node_type: str = "object",
        relations: Optional[List[dict]] = None,
        properties: dict = None,
        scope: str = None,
    ) -> str:
        """Upsert a spatial record and replace only supplied relation families.

        A physical-location update replaces the prior physical parent, while a
        scene/state-only update preserves that parent. Incoming child edges are
        always retained. If the object does not exist, it is created.
        """
        normalized_relations = []
        for rel in relations or []:
            target = rel.get("target", "").strip()
            rel_type = rel.get("type", "related").strip()
            if target:
                normalized_relations.append((target, rel_type))
        nid = self.spatial.add_node(
            name=name,
            node_type=node_type,
            properties=properties or {},
            step=self.current_step,
            scope=scope,
        )
        if self.spatial.relation_aware_updates:
            self.spatial.clear_outgoing_relation_families(
                nid,
                [rel_type for _, rel_type in normalized_relations],
            )
        else:
            self.spatial.clear_outgoing_edges(nid)
        for target, rel_type in normalized_relations:
            tid = self.spatial.add_node(
                name=target,
                node_type="object",
                properties={},
                step=self.current_step,
                scope=scope,
            )
            self.spatial.add_edge(nid, tid, rel_type)
        return nid

    def add_event(
        self,
        action: str,
        success: bool,
        feedback: str = "",
        note: str = "",
        event_summary: str = "",
        scene_ref: Optional[str] = None,
        spatial_refs: Optional[List[str]] = None,
        metadata: dict = None,
    ) -> str:
        """写入事件记忆，自动关联动作中提到的空间节点。"""
        # 从 action 文本中提取已知空间节点 ID
        auto_refs = [
            oid
            for name, ids in self.spatial.name_index.items()
            if name in action.lower()
            for oid in ids
        ]
        merged_refs = list({*(spatial_refs or []), *auto_refs})

        eid = self.event.add_event(
            step=self.current_step,
            action=action,
            success=success,
            feedback=feedback,
            event_summary=event_summary or note,
            scene_ref=scene_ref,
            spatial_refs=merged_refs,
            metadata=metadata,
        )
        self.experience.observe_event(self.event.events[-1])
        for ref in merged_refs:
            if ref in self.spatial.nodes:
                if eid not in self.spatial.nodes[ref]["related_event"]:
                    self.spatial.nodes[ref]["related_event"].append(eid)
        return eid

    def add_scene(
        self,
        caption: str,
        visible_objects: Optional[List[str]] = None,
        image_path: Optional[str] = None,
        scope: str = None,
    ) -> str:
        """写入场景快照，并为可见物体建立 scene 引用。
        image_path 未传时自动使用 current_image_path（由 planner.update() 注入）。
        """
        visible_objects = visible_objects or []
        sid = self.scene.add_scene(
            caption=caption,
            step=self.current_step,
            image_path=image_path or self.current_image_path,
            visible_objects=visible_objects,
            scope=scope,
        )
        for obj in visible_objects:
            for oid in self.spatial.name_index.get(obj.lower(), []):
                if self.spatial.nodes[oid].get("scope") != scope:
                    continue
                if sid not in self.spatial.nodes[oid]["related_scene"]:
                    self.spatial.nodes[oid]["related_scene"].append(sid)
        return sid

    # ── 查询接口 ──

    def query_spatial(
        self,
        query: str,
        top_k: int = 3,
        scope: str = None,
    ) -> List[dict]:
        top_k = min(12, max(0, top_k))
        if not top_k:
            return []
        results = self.spatial.query(query, top_k, scope=scope)
        for r in results:
            r["_layer"] = "spatial"
        return results

    def query_event(self, query: str, top_k: int = 3) -> List[dict]:
        top_k = min(12, max(0, top_k))
        if not top_k:
            return []
        results = self.event.query(query, top_k)
        for r in results:
            r["_layer"] = "event"
        return results

    def query_scene(self, query: str, top_k: int = 3, scope: str = None) -> List[dict]:
        top_k = min(12, max(0, top_k))
        if not top_k:
            return []
        results = self.scene.query(query, top_k, scope=scope)
        for r in results:
            r["_layer"] = "scene"
        return results

    def query_experience(
        self, query: str, top_k: int = 3, namespace: str = None
    ) -> List[dict]:
        top_k = min(12, max(0, top_k))
        if not top_k:
            return []
        results = self.experience.query(query, top_k=top_k, namespace=namespace)
        for r in results:
            r["_layer"] = "experience"
        return results

    def query_habits(
        self, query: str, top_k: int = 3, namespace: str = None
    ) -> List[dict]:
        """Backward-compatible alias for corrected-placement experience retrieval."""
        results = self.experience.query_corrected_placements(
            query,
            top_k=top_k,
            namespace=namespace,
        )
        for r in results:
            r["_layer"] = "experience"
        return results

    def query_locations(
        self, query: str, top_k: int = 3, namespace: str = None
    ) -> List[dict]:
        """Retrieve current object-location facts from consolidated experience memory."""
        results = self.experience.query_locations(
            query,
            top_k=top_k,
            namespace=namespace,
        )
        for r in results:
            r["_layer"] = "experience"
        return results

    def query_constraints(
        self, query: str, top_k: int = 3, namespace: str = None
    ) -> List[dict]:
        """Retrieve consolidated interaction constraints and positive affordances."""
        results = self.experience.query_interaction_knowledge(
            query,
            top_k=top_k,
            namespace=namespace,
        )
        for r in results:
            r["_layer"] = "experience"
        return results

    def query_interaction_states(
        self,
        query: str,
        top_k: int = 3,
        namespace: str = None,
    ) -> List[dict]:
        """Retrieve conflict-resolved interaction states and action policies."""
        results = self.experience.query_interaction_states(
            query,
            top_k=top_k,
            namespace=namespace,
        )
        for r in results:
            r["_layer"] = "experience"
        return results

    def query_experience_portfolio(
        self,
        query: str,
        top_k: int = 3,
        namespace: str = None,
    ) -> List[dict]:
        """Retrieve a typed portfolio of consolidated memories."""
        results = self.experience.query_portfolio(
            query,
            top_k=top_k,
            namespace=namespace,
        )
        for r in results:
            r["_layer"] = "experience"
        return results

    def query_relevant(
        self,
        query: str,
        *,
        namespace: str = None,
        experience_top_k: int = 3,
        event_top_k: int = 5,
        spatial_top_k: int = 5,
        scene_top_k: int = 2,
    ) -> List[dict]:
        """Unified retrieval with consolidated experience as the first-class layer.

        If an experience matches, return that consolidated memory first.
        Habit memories carry their own sanitized semantic support traces, so
        raw support events are intentionally not mixed back into habit results.
        Other memory types still include support events as low-level evidence.
        """
        query_intent = self.experience.infer_query_intent(query)
        if query_intent == "habit":
            experiences = self.query_habits(
                query,
                top_k=experience_top_k,
                namespace=namespace,
            )
        elif query_intent == "location":
            experiences = self.query_locations(
                query,
                top_k=experience_top_k,
                namespace=namespace,
            )
        elif query_intent == "constraint":
            experiences = self.query_constraints(
                query,
                top_k=experience_top_k,
                namespace=namespace,
            )
        elif query_intent == "mixed":
            experiences = self.query_experience_portfolio(
                query,
                top_k=experience_top_k,
                namespace=namespace,
            )
        else:
            experiences = self.query_experience_portfolio(
                query,
                top_k=experience_top_k,
                namespace=namespace,
            )
        if experiences:
            if query_intent == "habit":
                return experiences
            support_ids = {
                event_id
                for item in experiences
                for event_id in item.get("source_event_ids", [])
            }
            support_events = [
                {**event, "_layer": "event", "support_for_experience": True}
                for event in self.event.events
                if event.get("id") in support_ids
            ]
            return [*experiences, *support_events[:event_top_k]]

        return [
            *self.query_event(query, top_k=event_top_k),
            *self.query_spatial(query, top_k=spatial_top_k),
            *self.query_scene(query, top_k=scene_top_k),
        ]

    # ── 辅助 ──

    def dump_object_list_text(self) -> str:
        """返回所有已记录物体的名称列表，用于上下文压缩后防止重复记录。"""
        if not self.spatial.nodes:
            return "[Recorded Objects]\n(none)"
        names = [node["name"] for node in self.spatial.nodes.values()]
        return "[Recorded Objects]\n" + "\n".join(f"  - {n}" for n in names)

    def get_agent_state(self) -> dict:
        held_objects = [
            node["name"]
            for node in self.spatial.nodes.values()
            if node.get("properties", {}).get("held_by_agent")
        ]
        return {
            "held_objects": held_objects,
            "known_object_count": len(self.spatial.nodes),
            "event_count": len(self.event.events),
        }

    def get_stats(self) -> dict:
        return {
            "spatial_nodes": len(self.spatial.nodes),
            "spatial_edges": len(self.spatial.edges),
            "scene_count": len(self.scene.scenes),
            "event_count": len(self.event.events),
            "experience_count": len(self.experience.experiences),
            "consolidated_experience_count": len(self.experience.consolidated),
            "consolidated_habit_pair_count": len(self.experience.habit_pairs),
            "consolidated_habit_object_belief_count": len(
                self.experience.habit_object_beliefs
            ),
            "interaction_failure_count": len(self.experience.interaction_failures),
            "consolidated_failure_constraint_count": len(
                self.experience.failure_constraints
            ),
            "interaction_affordance_observation_count": len(
                self.experience.interaction_affordance_observations
            ),
            "consolidated_interaction_affordance_count": len(
                self.experience.interaction_affordances
            ),
            "consolidated_interaction_state_count": len(
                self.experience.interaction_states
            ),
            "location_observation_count": len(self.experience.location_observations),
            "consolidated_location_count": len(self.experience.latest_locations),
            "current_step": self.current_step,
        }
