import heapq
import itertools

from algorithms.registry import AlgorithmRegistry
from core.base_algorithm import BaseGraphAlgorithm
from core.graph_model import SpatialGraphModel


class StaticPrim(BaseGraphAlgorithm):
    SUPPORTED_EVENT_TYPES = frozenset({"DYNAMIC_EDGE_WEIGHT"})

    def __init__(self, model: SpatialGraphModel):
        super().__init__(model)
        self.in_tree = set()

    def run(self, start_node=None, goal_node=None) -> tuple[list, float]:
        """start_node: simpul awal (atau list/set simpul untuk himpunan V).
        goal_node diabaikan, hanya ada agar signature sama dengan Dijkstra.
        """
        mst_edges = []
        total_cost = 0.0
        self.in_tree = set()

        # Construction B bekerja pada graf tak berarah
        g = self.graph.to_undirected()
        nodes = list(g.nodes())
        if not nodes:
            return mst_edges, total_cost

        # Himpunan V awal (nonempty, arbitrary but fixed)
        if start_node is None or start_node not in g:
            if isinstance(start_node, (list, set, tuple, frozenset)):
                initial = [n for n in start_node if n in g] or [nodes[0]]
            else:
                initial = [nodes[0]]
        else:
            initial = [start_node]

        counter = itertools.count()  # tie-breaker agar node tidak dibandingkan
        pq = []

        def add_vertex(node):
            """Masukkan simpul ke pohon, lalu dorong semua edge yang
            menghubungkannya ke simpul di luar pohon."""
            self.in_tree.add(node)
            for nbr, data in g[node].items():
                if nbr not in self.in_tree:
                    w = float(data.get("weight", 1.0))
                    heapq.heappush(pq, (w, next(counter), node, nbr))

        for node in initial:
            add_vertex(node)

        self.log_step(
            event_type="INITIALIZATION",
            state_mutations={
                "start_nodes": list(initial),
                "candidate_edges": len(pq),
            },
        )

        while True:
            while pq:
                weight, _, u, v = heapq.heappop(pq)

                # Kedua ujung sudah di pohon: edge ini akan membentuk loop
                if v in self.in_tree:
                    continue

                add_vertex(v)
                mst_edges.append((u, v, weight))
                total_cost += weight

                self.log_step(
                    event_type="EDGE_ADDED",
                    state_mutations={
                        "added_edge": [u, v],
                        "weight": weight,
                        "current_mst_count": len(mst_edges),
                        "total_cost": total_cost,
                        "tree_size": len(self.in_tree),
                    },
                )

            remaining = next((n for n in nodes if n not in self.in_tree), None)
            if remaining is None:
                break
            add_vertex(remaining)

        return mst_edges, total_cost

    def handle_dynamic_event(self, event_data: dict):
        edge = event_data["edge"]
        new_weight = event_data["new_weight"]
        old_weight = self.graph[edge[0]][edge[1]].get("weight", 1)

        self.graph[edge[0]][edge[1]]["weight"] = new_weight

        self.log_step(
            event_type="UPDATE_WEIGHT_EVENT",
            state_mutations={
                "target_edge": list(edge),
                "old_weight": old_weight,
                "new_weight": new_weight,
            },
        )

        return self.run(event_data.get("start"), event_data.get("goal"))


AlgorithmRegistry.register("static_prim", StaticPrim)
AlgorithmRegistry.register_alias("prim", "static_prim")