"""
backend/memory_graph.py — 记忆拓扑图 + LM-Nav 式图搜索（P3）

借鉴 LM-Nav：把走过的成功轨迹沉淀成一张 **2D 拓扑图**，下次遇到相似指令时，
先把指令里的地标对齐到图节点（grounding 打分），再用图搜索（Dijkstra）把
起点→各地标节点串成一条"熟路"，优先沿熟路走，省去重复探索。

与 DisasterClaw 适配：
    - 节点 = UAV 实际飞过/观测过的位置（按 merge_radius_m 阈值合并邻近点），
      节点摘要累积该处看到的目标类别 / risk / 到过的指令 & 地标标签。
    - 边 = 轨迹上相邻节点，权重 = 实际地表距离（米）。
    - 跨 episode / 跨任务持久化为 JSON（save/load），实现"越用越熟"。

依赖：仅 geo.latlon_to_meters 做距离；不碰模型 / socket / IO 之外的东西。
grounding 打分用注入的 scorer(node, phrase)->[0,1]，默认提供字符 bigram 文本匹配，
也可换成 VLM/CLIP。便于单测。
"""

from __future__ import annotations

import heapq
import json
import math
import time
from pathlib import Path
from typing import Callable, Optional

from geo import latlon_to_meters

# scorer(node_dict, phrase) -> 匹配分 ∈ [0,1]
Scorer = Callable[[dict, str], float]


def _dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    n, e = latlon_to_meters(lat1, lon1, lat2, lon2)
    return math.hypot(n, e)


# ── 默认文本 scorer：字符 bigram Jaccard ────────────────────────────────
def _bigrams(s: str) -> set[str]:
    s = "".join((s or "").split())
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else ({s} if s else set())


def _bigram_jaccard(a: str, b: str) -> float:
    ba, bb = _bigrams(a), _bigrams(b)
    if not ba or not bb:
        return 0.0
    inter = len(ba & bb)
    union = len(ba | bb)
    return inter / union if union else 0.0


def node_text(node: dict) -> str:
    """节点用于匹配的文本：目标类别 + 到过的地标 + 指令片段。"""
    parts: list[str] = []
    parts += list((node.get("labels") or {}).keys())
    parts += list(node.get("landmarks") or [])
    parts += list(node.get("instructions") or [])
    return " ".join(parts)


def text_match_scorer(node: dict, phrase: str) -> float:
    """默认 grounding 打分：地标短语与节点文本的最大 bigram 相似度。

    对节点每个标签单独比一次取最大（"完全损毁建筑" vs "完全损毁的建筑" 仍高分），
    再与整段节点文本比一次，取最大。
    """
    if not phrase:
        return 0.0
    best = _bigram_jaccard(phrase, node_text(node))
    for label in (node.get("labels") or {}).keys():
        best = max(best, _bigram_jaccard(phrase, label))
    for lm in (node.get("landmarks") or []):
        best = max(best, _bigram_jaccard(phrase, lm))
    return round(best, 4)


class MemoryGraph:
    """跨任务持久化的 2D 拓扑记忆图。"""

    def __init__(self, merge_radius_m: float = 15.0):
        self.merge_radius_m = float(merge_radius_m)
        self._nodes: dict[int, dict] = {}
        self._edges: dict[int, dict[int, float]] = {}  # 无向：两侧都存
        self._next_id = 0

    # ── 构建 ─────────────────────────────────────────────────────────
    def _nearest(self, lat: float, lon: float) -> tuple[Optional[int], float]:
        best_id, best_d = None, float("inf")
        for nid, nd in self._nodes.items():
            d = _dist_m(lat, lon, nd["lat"], nd["lon"])
            if d < best_d:
                best_id, best_d = nid, d
        return best_id, best_d

    def add_waypoint(
        self,
        lat: float,
        lon: float,
        alt: float = 0.0,
        labels: Optional[dict] = None,
        risk: str = "none",
        summary: str = "",
    ) -> int:
        """加入一个轨迹点；落在 merge_radius 内则并入最近节点，否则新建。返回节点 id。"""
        nid, d = self._nearest(lat, lon)
        if nid is not None and d <= self.merge_radius_m:
            nd = self._nodes[nid]
            nd["visits"] += 1
            nd["last_ts"] = time.time()
            self._merge_labels(nd, labels)
            if risk and risk != "none":
                nd["risks"][risk] = nd["risks"].get(risk, 0) + 1
            if summary:
                nd["summaries"] = (nd.get("summaries") or [])[-4:] + [summary]
            return nid

        new_id = self._next_id
        self._next_id += 1
        self._nodes[new_id] = {
            "id": new_id,
            "lat": float(lat),
            "lon": float(lon),
            "alt": float(alt),
            "labels": dict(labels or {}),
            "risks": ({risk: 1} if risk and risk != "none" else {}),
            "landmarks": [],
            "instructions": [],
            "summaries": ([summary] if summary else []),
            "visits": 1,
            "last_ts": time.time(),
        }
        self._edges.setdefault(new_id, {})
        return new_id

    @staticmethod
    def _merge_labels(nd: dict, labels: Optional[dict]) -> None:
        if not labels:
            return
        bucket = nd.setdefault("labels", {})
        for k, v in labels.items():
            bucket[k] = bucket.get(k, 0) + int(v)

    def connect(self, a: int, b: int) -> None:
        if a == b or a not in self._nodes or b not in self._nodes:
            return
        d = _dist_m(
            self._nodes[a]["lat"], self._nodes[a]["lon"],
            self._nodes[b]["lat"], self._nodes[b]["lon"],
        )
        self._edges.setdefault(a, {})[b] = d
        self._edges.setdefault(b, {})[a] = d

    def add_trajectory(
        self,
        points: list[dict],
        instruction: str = "",
        landmarks: Optional[list[str]] = None,
        success: bool = True,
    ) -> list[int]:
        """把一条轨迹（依次的观测点）写入图：合并成节点 + 相邻连边 + 终点打地标标签。

        points: [{lat, lon, alt, labels:{cls:count}, risk, summary}, ...]
        """
        ids: list[int] = []
        prev: Optional[int] = None
        for p in points:
            nid = self.add_waypoint(
                lat=float(p["lat"]), lon=float(p["lon"]), alt=float(p.get("alt", 0.0)),
                labels=p.get("labels"), risk=str(p.get("risk", "none")),
                summary=str(p.get("summary", "")),
            )
            if prev is not None and prev != nid:
                self.connect(prev, nid)
            ids.append(nid)
            prev = nid
        # 成功到达：给终点（最后一个节点）打上指令 / 地标标签，便于后续匹配。
        if success and ids:
            term = self._nodes[ids[-1]]
            if instruction and instruction not in term["instructions"]:
                term["instructions"] = (term["instructions"] + [instruction])[-5:]
            for lm in (landmarks or []):
                if lm and lm not in term["landmarks"]:
                    term["landmarks"].append(lm)
        return ids

    # ── 查询 / 搜索 ──────────────────────────────────────────────────
    def match_nodes(
        self, phrase: str, scorer: Scorer = text_match_scorer, top_k: int = 5
    ) -> list[tuple[dict, float]]:
        scored = [(nd, scorer(nd, phrase)) for nd in self._nodes.values()]
        scored = [x for x in scored if x[1] > 0.0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def shortest_path(self, src: int, dst: int) -> Optional[list[int]]:
        """Dijkstra（边权=米）。不可达返回 None。"""
        if src not in self._nodes or dst not in self._nodes:
            return None
        if src == dst:
            return [src]
        dist = {src: 0.0}
        prev: dict[int, int] = {}
        pq: list[tuple[float, int]] = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                break
            if d > dist.get(u, float("inf")):
                continue
            for v, w in self._edges.get(u, {}).items():
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if dst not in dist:
            return None
        path = [dst]
        while path[-1] != src:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    def plan(
        self,
        landmarks: list[str],
        start_lat: float,
        start_lon: float,
        scorer: Scorer = text_match_scorer,
        min_score: float = 0.34,
        start_attach_m: float = 60.0,
    ) -> Optional[dict]:
        """LM-Nav 式规划：为（最终）地标找最佳记忆节点，串出一条 walk。

        - 用最终地标（landmarks[-1]）匹配目标节点；分数 < min_score → 返回 None（交回探索）。
        - 起点若靠近某节点（≤ start_attach_m）→ Dijkstra 走熟路；否则直飞目标节点。
        返回 {waypoints:[{lat,lon,alt}], target_score, target_label, node_ids, mode}。
        """
        if not self._nodes or not landmarks:
            return None
        goal_phrase = landmarks[-1]
        matches = self.match_nodes(goal_phrase, scorer, top_k=1)
        if not matches or matches[0][1] < min_score:
            return None
        goal_node, score = matches[0]
        goal_id = goal_node["id"]

        src_id, src_d = self._nearest(start_lat, start_lon)
        node_ids: list[int]
        mode: str
        if src_id is not None and src_d <= start_attach_m:
            path = self.shortest_path(src_id, goal_id)
            if path:
                node_ids, mode = path, "graph_walk"
            else:
                node_ids, mode = [goal_id], "direct"
        else:
            node_ids, mode = [goal_id], "direct"

        waypoints = [
            {
                "lat": self._nodes[i]["lat"],
                "lon": self._nodes[i]["lon"],
                "alt": self._nodes[i]["alt"],
            }
            for i in node_ids
        ]
        return {
            "waypoints": waypoints,
            "node_ids": node_ids,
            "target_score": round(score, 4),
            "target_label": goal_phrase,
            "mode": mode,
        }

    # ── 统计 / 持久化 ────────────────────────────────────────────────
    def stats(self) -> dict:
        edges = sum(len(v) for v in self._edges.values()) // 2
        return {
            "nodes": len(self._nodes),
            "edges": edges,
            "merge_radius_m": self.merge_radius_m,
        }

    def to_dict(self) -> dict:
        return {
            "merge_radius_m": self.merge_radius_m,
            "next_id": self._next_id,
            "nodes": self._nodes,
            "edges": {str(k): v for k, v in self._edges.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryGraph":
        g = cls(merge_radius_m=float(data.get("merge_radius_m", 15.0)))
        g._next_id = int(data.get("next_id", 0))
        g._nodes = {int(k): v for k, v in (data.get("nodes") or {}).items()}
        g._edges = {
            int(k): {int(nk): float(nv) for nk, nv in (v or {}).items()}
            for k, v in (data.get("edges") or {}).items()
        }
        for nid in g._nodes:
            g._edges.setdefault(nid, {})
        if g._nodes and g._next_id <= max(g._nodes):
            g._next_id = max(g._nodes) + 1
        return g

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path, merge_radius_m: float = 15.0) -> "MemoryGraph":
        path = Path(path)
        if not path.exists():
            return cls(merge_radius_m=merge_radius_m)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls.from_dict(data)
        except Exception:
            return cls(merge_radius_m=merge_radius_m)
