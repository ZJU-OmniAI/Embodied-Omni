"""
事件记忆：按时间顺序记录 agent 每步的动作、结果与环境反馈，只追加不修改。

检索分两级（均在本模块内完成）：
  Level 1  _keyword_search()      关键词匹配，速度快
  Level 2  _embedding_search()    embedding 语义搜索，关键词不足时补齐

EmbeddingEngine 由 MemorySystem 创建后注入，所有记忆模块共享同一实例。

写入：update_info() → memory.memory_add("event", {...})
读取：build_memory_context() → event.to_text(query=instruction)
"""

import numpy as np
from typing import List


class EventMemory:
    """
    事件列表，每条记录包含：
        id, step, action, success, feedback, event_summary,
        scene_ref（未填充）, spatial_ref（未填充）
    """

    def __init__(self, config: dict = None, embedding_engine=None):
        config = config or {}
        self.max_events = config.get("max_events", 1000)
        self.events: List[dict] = []
        self._emb = embedding_engine  # 共享 EmbeddingEngine，可为 None
        self._emb_cache: dict = {}  # event_id -> np.ndarray

    def reset(self):
        self.events.clear()
        self._emb_cache.clear()

    def add_event(
        self,
        step: int,
        action: str,
        success: bool,
        feedback: str = "",
        event_summary: str = "",
        scene_ref: str = None,
        spatial_refs: List[str] = None,
        metadata: dict = None,
    ) -> str:
        """追加一条事件，超出 max_events 时滚动丢弃最旧的。"""
        eid = f"event_{len(self.events)}"
        event = {
            "id": eid,
            "step": step,
            "action": action,
            "success": success,
            "feedback": feedback,
            "event_summary": event_summary,
            "scene_ref": scene_ref,
            "spatial_ref": spatial_refs or [],
            "metadata": metadata or {},
        }
        self.events.append(event)
        if len(self.events) > self.max_events:
            oldest = self.events.pop(0)
            self._emb_cache.pop(oldest["id"], None)
        return eid

    # ── 两级检索 ──

    def _keyword_search(self, query: str, top_k: int) -> List[dict]:
        """Level 1：在 action / event_summary 中匹配 query，最新优先。"""
        kw = query.lower()
        results = []
        for event in reversed(self.events):
            if (
                kw in event["action"].lower()
                or kw in event.get("event_summary", "").lower()
            ):
                results.append(event)
                if len(results) >= top_k:
                    break
        return results

    def _embedding_search(self, query: str, top_k: int, exclude: set) -> List[dict]:
        """Level 2：embedding 余弦相似度搜索，跳过已在 exclude 中的事件。"""
        if self._emb is None or not self.events:
            return []

        # 增量计算缺失的 embedding
        for event in self.events:
            eid = event["id"]
            if eid not in self._emb_cache:
                # 用这些文本计算 embedding 是否合适?
                text = f"{event['action']} {event.get('event_summary', '')} {event.get('feedback', '')}"
                try:
                    self._emb_cache[eid] = self._emb.encode_single(text)
                except (ImportError, ModuleNotFoundError):
                    self._emb = None
                    return []

        candidates = [
            (e, self._emb_cache[e["id"]]) for e in self.events if e["id"] not in exclude
        ]
        if not candidates:
            return []

        try:
            query_emb = self._emb.encode_single(query)
        except (ImportError, ModuleNotFoundError):
            self._emb = None
            return []
        emb_matrix = np.array([c[1] for c in candidates])
        scores = self._emb.batch_cosine_similarity(query_emb, emb_matrix)

        results = []
        for idx in np.argsort(scores)[::-1]:
            if scores[idx] > 0.1:
                results.append(candidates[idx][0])
                if len(results) >= top_k:
                    break
        return results

    def query(self, query: str, top_k: int = 5) -> List[dict]:
        """两级检索：关键词优先，不足时 embedding 补齐。"""
        results = self._keyword_search(query, top_k)
        if len(results) < top_k:
            exclude = {r["id"] for r in results}
            results += self._embedding_search(query, top_k - len(results), exclude)
        return results[:top_k]

    # ── 模式检测（与检索逻辑无关，供 build_memory_context 生成警告） ──

    def get_action_pattern(self) -> dict:
        """
        检测最近 10 步内的异常模式：
          - repeated_failures：同一动作失败 ≥ 2 次
          - loop_detected：最近 3 步动作序列 == 前 3 步
        """
        if len(self.events) < 3:
            return {"repeated_failures": [], "loop_detected": False}

        recent = self.events[-10:]

        failure_counts = {}
        for e in recent:
            if not e["success"]:
                a = e["action"]
                failure_counts[a] = failure_counts.get(a, 0) + 1

        repeated_failures = [a for a, c in failure_counts.items() if c >= 2]

        loop_detected = False
        if len(recent) >= 6:
            last3 = [e["action"] for e in recent[-3:]]
            prev3 = [e["action"] for e in recent[-6:-3]]
            loop_detected = last3 == prev3

        return {"repeated_failures": repeated_failures, "loop_detected": loop_detected}

    # ── Prompt 生成 ──

    def to_text(self, query: str = "", top_k: int = 5) -> str:
        """
        生成注入 prompt 的事件文本。
        query 非空时走两级检索；为空时退化为取最新 top_k 条。
        警告文本由 build_memory_context() 统一生成，此处不重复输出。
        """
        if not self.events:
            return "No event memory recorded yet."

        results = (
            self.query(query, top_k) if query else list(reversed(self.events[-top_k:]))
        )

        lines = ["[Event Memory]"]
        for event in results:
            status = "SUCCESS" if event["success"] else "FAILED"
            lines.append(f"- Step {event['step']}: {event['action']} [{status}]")
            if event.get("feedback"):
                lines.append(f"  Feedback: {event['feedback']}")

        return "\n".join(lines)
