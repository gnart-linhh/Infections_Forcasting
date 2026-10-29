"""Exact L1 neighbors with deterministic (distance, source row) tie breaking."""
import numpy as np
from scipy.spatial import cKDTree


class ExactNeighbors:
    def __init__(self, x, workers=1):
        self.x = np.asarray(x, dtype=float)
        self.workers = workers
        self.tree = cKDTree(self.x) if len(self.x) >= 10000 else None

    @staticmethod
    def _ordered(distances, ids, count):
        # Only sort the selected set. Stable argsort of the entire matrix is
        # unnecessary, but tied boundary rows must keep the original row order.
        if len(ids) > count:
            cutoff = np.partition(distances, count - 1)[count - 1]
            below = np.flatnonzero(distances < cutoff)
            equal = np.flatnonzero(distances == cutoff)
            equal = equal[np.argsort(ids[equal], kind='stable')[:count-len(below)]]
            take = np.r_[below, equal]
            distances, ids = distances[take], ids[take]
        order = np.lexsort((ids, distances))
        return ids[order], distances[order]

    def query(self, queries, count):
        queries = np.atleast_2d(queries)
        count = min(int(count), len(self.x))
        if count < 1:
            raise ValueError('Need positive neighbor count and nonempty embeddings')
        if self.tree is None:
            results = [self._ordered(np.abs(self.x-q).sum(axis=1), np.arange(len(self.x)), count) for q in queries]
        else:
            distances, indices = self.tree.query(queries, k=min(count+1, len(self.x)), p=1, eps=0,
                                                 workers=self.workers)
            results = []
            for query, dists, ids in zip(queries, np.atleast_2d(distances), np.atleast_2d(indices)):
                if len(ids) > count and dists[count-1] == dists[count]:
                    ids = np.asarray(self.tree.query_ball_point(query, np.nextafter(dists[count], np.inf), p=1), dtype=int)
                actual_distances = np.abs(self.x[ids]-query).sum(axis=1)
                results.append(self._ordered(actual_distances, ids, count))
        return np.array([r[0] for r in results]), np.array([r[1] for r in results])
