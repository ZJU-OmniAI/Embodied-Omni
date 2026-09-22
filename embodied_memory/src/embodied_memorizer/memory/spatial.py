"""
空间记忆：基于图结构存储空间/物体信息。

节点表示物体、房间或地标；边表示空间关系（包含、相邻、可达等）。
"""

import re
import uuid
import numpy as np
from typing import Dict, List, Optional, Tuple


_PHYSICAL_LOCATION_RELATIONS = frozenset(
    {
        "at",
        "held_by_agent",
        "in",
        "inside",
        "located_in",
        "on",
        "on_or_in",
    }
)

_ALL_SCOPES = object()


def _normalize_scope(scope: object) -> Optional[str]:
    if scope is None:
        return None
    normalized = str(scope).strip()
    return normalized or None


def _relation_family(relation: str) -> str:
    normalized = str(relation or "related").strip().lower()
    if normalized in _PHYSICAL_LOCATION_RELATIONS:
        return "physical_location"
    if normalized == "seen_in":
        return "scene_visibility"
    return normalized


def _incoming_relation(relation: str) -> str:
    normalized = str(relation or "related").strip().lower()
    inverse = {
        "held_by_agent": "holding",
        "in": "contains",
        "inside": "contains",
        "on": "supports",
        "on_or_in": "contains",
        "seen_in": "contains",
    }
    return inverse.get(normalized, f"incoming_{normalized}")


class SpatialMemory:
    """
    基于图结构的空间记忆。节点表示物体/房间/地标，边表示空间关系（包含、相邻、可达等）。
    """

    def __init__(self, config: dict = None, embedding_engine=None):
        config = config or {}
        self.max_nodes = config.get("max_nodes", 500)  # 是否有必要？
        self.relation_aware_updates = bool(config.get("relation_aware_updates", True))
        self.directional_incoming_relations = bool(
            config.get("directional_incoming_relations", True)
        )

        self.nodes: Dict[str, dict] = {}  # node_id -> node_data
        self.edges: Dict[Tuple[str, str], dict] = {}  # (id1, id2) -> edge_data
        self.name_index: Dict[
            str, List[str]
        ] = {}  # name -> [node_ids] ，可能存在同名物体？
        self._emb = embedding_engine  # 共享 EmbeddingEngine，可为 None
        self._emb_cache: dict = {}  # node_id -> np.ndarray

    def reset(self):
        self.nodes.clear()
        self.edges.clear()
        self.name_index.clear()
        self._emb_cache.clear()

    def add_node(
        self,
        name: str,
        node_type: str = "object",
        properties: dict = None,
        node_id: str = None,
        step: int = None,
        scope: str = None,
        related_scene: List[str] = None,
        related_event: List[str] = None,
    ) -> str:
        """添加或更新空间节点，返回 node_id。同名同 scope 节点存在时更新第一个。"""
        normalized_scope = _normalize_scope(scope)
        existing_ids = [
            nid
            for nid in self.name_index.get(name.lower(), [])
            if _normalize_scope(self.nodes[nid].get("scope")) == normalized_scope
        ]
        if existing_ids and node_id is None:
            nid = existing_ids[0]
            node = self.nodes[nid]
            if properties is not None:
                node["properties"].update(properties)
            return nid

        nid = node_id or f"spatial_{uuid.uuid4().hex[:8]}"
        self.nodes[nid] = {
            "type": node_type,
            "name": name,
            "properties": properties or {},
            "step": step,
            "scope": normalized_scope,
            "related_scene": related_scene or [],  # 关联的场景快照 ID 列表
            "related_event": related_event or [],  # 关联的事件 ID 列表
        }
        self.name_index.setdefault(name.lower(), []).append(nid)
        return nid

    def remove_node(self, node_id: str) -> bool:
        """删除节点及其所有关联边，并更新名称索引。"""
        if node_id not in self.nodes:
            return False
        name = self.nodes[node_id]["name"].lower()
        if name in self.name_index:
            self.name_index[name] = [n for n in self.name_index[name] if n != node_id]
            if not self.name_index[name]:
                del self.name_index[name]
        # 删除所有包含该节点的边
        self.edges = {k: v for k, v in self.edges.items() if node_id not in k}
        del self.nodes[node_id]
        return True

    def update_node(self, node_id: str, updates: dict) -> bool:
        """更新节点字段；properties 字段执行合并更新而非覆盖。"""
        if node_id not in self.nodes:
            return False
        for key, value in updates.items():
            if key == "properties" and isinstance(value, dict):
                self.nodes[node_id]["properties"].update(value)
            else:
                self.nodes[node_id][key] = value
        return True

    def clear_node_edges(self, node_id: str):
        """删除与 node_id 相连的所有边（用于原子性关系替换）。"""
        self.edges = {k: v for k, v in self.edges.items() if node_id not in k}

    def clear_outgoing_edges(self, node_id: str):
        """删除 node_id 的出边，同时保留其他节点指向它的入边。"""
        self.edges = {k: v for k, v in self.edges.items() if k[0] != node_id}

    def clear_outgoing_relation_families(
        self,
        node_id: str,
        relations: List[str],
    ) -> None:
        """Replace only outgoing relation families explicitly present in an update."""
        families = {_relation_family(relation) for relation in relations}
        if not families:
            return
        self.edges = {
            key: value
            for key, value in self.edges.items()
            if not (
                key[0] == node_id
                and _relation_family(value.get("relation")) in families
            )
        }

    def add_edge(
        self,
        node_id1: str,
        node_id2: str,
        relation: str = "reachable",
        distance: float = 0.0,
    ):
        """在两个已存在的节点之间添加有向边。"""
        if node_id1 in self.nodes and node_id2 in self.nodes:
            self.edges[(node_id1, node_id2)] = {
                "relation": relation,
                "distance": distance,
            }

    def _node_matches_scope(self, node_id: str, scope: object) -> bool:
        if scope is _ALL_SCOPES:
            return True
        return _normalize_scope(self.nodes[node_id].get("scope")) == _normalize_scope(
            scope
        )

    def _query_by_name(self, name: str, scope: object = _ALL_SCOPES) -> List[dict]:
        """按名称模糊匹配节点（大小写不敏感）。"""
        results = []
        name_lower = name.lower()
        query_tokens = set(re.findall(r"[a-z0-9_]+", name_lower))
        matching_names = []
        for order, (key, ids) in enumerate(self.name_index.items()):
            if name_lower not in key and key not in name_lower:
                continue
            if key == name_lower:
                priority = 0
            elif key in query_tokens:
                priority = 1
            else:
                priority = 2
            matching_names.append((priority, -len(key), order, ids))
        for _, _, _, ids in sorted(matching_names):
            for nid in ids:
                if not self._node_matches_scope(nid, scope):
                    continue
                results.append({"id": nid, **self.nodes[nid]})
        return results

    def _keyword_search(
        self,
        query: str,
        top_k: int,
        exclude: set,
        scope: object = _ALL_SCOPES,
    ) -> List[dict]:
        """关键词检索：对 query 整体及每个词分别匹配节点名称，跳过 exclude 中的节点。"""
        results = []
        seen = set(exclude)
        for term in [query] + [w for w in query.lower().split() if len(w) > 2]:
            for r in self._query_by_name(term, scope=scope):
                if r["id"] not in seen:
                    results.append(r)
                    seen.add(r["id"])
                    if len(results) >= top_k:
                        return results
        return results

    def _embedding_search(
        self,
        query: str,
        top_k: int,
        exclude: set,
        scope: object = _ALL_SCOPES,
    ) -> List[dict]:
        """Embedding 语义检索节点名称，跳过 exclude 中的节点。"""
        if self._emb is None or not self.nodes:
            return []
        eligible_ids = [
            nid
            for nid in self.nodes
            if nid not in exclude and self._node_matches_scope(nid, scope)
        ]
        if not eligible_ids:
            return []
        # 增量计算缺失的 embedding
        for nid in eligible_ids:
            node = self.nodes[nid]
            if nid not in self._emb_cache:
                self._emb_cache[nid] = self._emb.encode_single(node["name"])

        candidates = [
            (nid, self._emb_cache[nid])
            for nid in eligible_ids
            if nid in self._emb_cache
        ]
        if not candidates:
            return []

        query_emb = self._emb.encode_single(query)
        scores = self._emb.batch_cosine_similarity(
            query_emb, np.array([c[1] for c in candidates])
        )
        results = []
        for idx in np.argsort(scores)[::-1]:
            if scores[idx] <= 0.1:
                break
            nid = candidates[idx][0]
            results.append({"id": nid, **self.nodes[nid]})
            if len(results) >= top_k:
                break
        return results

    def query(self, query: str, top_k: int = 10, scope: str = None) -> List[dict]:
        """
        两阶段检索 + 边扩展：
          1. 关键词匹配节点名称
          2. Embedding 语义搜索补齐不足部分
          3. 沿边向外扩展一层，补充相关联节点
        """
        normalized_scope = _normalize_scope(scope)
        active_scope = _ALL_SCOPES if normalized_scope is None else normalized_scope

        def search(search_scope: object) -> List[dict]:
            matched = self._keyword_search(
                query,
                top_k,
                exclude=set(),
                scope=search_scope,
            )
            matched_ids = {r["id"] for r in matched}
            if len(matched) < top_k:
                embedded = self._embedding_search(
                    query,
                    top_k - len(matched),
                    exclude=matched_ids,
                    scope=search_scope,
                )
                matched += embedded
            return matched

        results = search(active_scope)
        if normalized_scope is not None and not results:
            active_scope = None
            results = search(active_scope)
        seen = {r["id"] for r in results}

        # 阶段 3：沿边扩展一层
        expanded = []
        for r in list(results):
            for n1, n2 in self.edges:
                neighbor_id = n2 if n1 == r["id"] else (n1 if n2 == r["id"] else None)
                if (
                    neighbor_id
                    and neighbor_id not in seen
                    and neighbor_id in self.nodes
                    and self._node_matches_scope(neighbor_id, active_scope)
                ):
                    expanded.append({"id": neighbor_id, **self.nodes[neighbor_id]})
                    seen.add(neighbor_id)

        # 为每个结果附加已解析的边信息（relation + target_name）
        all_results = results + expanded
        for r in all_results:
            edges = self._get_related_edges(r["id"], scope=active_scope)
            resolved = []
            for e in edges:
                target_id = e["to"]
                if target_id in self.nodes:
                    resolved.append(
                        {
                            "relation": e["relation"],
                            "target_name": self.nodes[target_id]["name"],
                        }
                    )
            r["_edges"] = resolved

        return all_results

    def _get_related_edges(
        self,
        node_id: str,
        scope: object = _ALL_SCOPES,
    ) -> List[dict]:
        """返回与指定节点相关的所有边（包括出边和入边）。"""
        results = []
        for (n1, n2), data in self.edges.items():
            if n1 == node_id:
                if not self._node_matches_scope(n2, scope):
                    continue
                results.append({"from": n1, "to": n2, **data})
            elif n2 == node_id:
                if not self._node_matches_scope(n1, scope):
                    continue
                relation = data.get("relation")
                if self.directional_incoming_relations:
                    relation = _incoming_relation(relation)
                results.append(
                    {
                        "from": n2,
                        "to": n1,
                        **data,
                        "relation": relation,
                        "source_relation": data.get("relation"),
                        "direction": "incoming",
                    }
                )
        return results

    def to_text(self, top_k: int = 10) -> str:
        """将空间记忆序列化为文本，用于注入 prompt。按插入顺序取最近 top_k 条。"""
        if not self.nodes:
            return "No spatial memory recorded yet."

        # 取最后插入的 top_k 个节点（dict 保持插入顺序）
        recent_nodes = list(self.nodes.items())[-top_k:]

        lines = ["[Spatial Memory]"]
        for nid, node in recent_nodes:
            props = node.get("properties", {})
            props_str = ""
            if props:
                props_items = [f"{k}={v}" for k, v in props.items()]
                props_str = f" [{', '.join(props_items)}]"

            lines.append(f"- {node['name']} ({node['type']}){props_str}")

            # 每个节点最多输出 3 条关联边，避免文本过长
            edges = self._get_related_edges(nid)
            for e in edges[:3]:
                other_id = e["to"]
                if other_id in self.nodes:
                    other_name = self.nodes[other_id]["name"]
                    lines.append(f"  -> {e['relation']} {other_name}")

        return "\n".join(lines)
