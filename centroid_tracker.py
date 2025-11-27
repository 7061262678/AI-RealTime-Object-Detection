from collections import OrderedDict
from typing import Dict, Tuple, List
import numpy as np

class CentroidTracker:
    def __init__(self, max_disappeared: int = 30):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        self.objects.pop(object_id, None)
        self.disappeared.pop(object_id, None)

    def update(self, rects: List[Tuple[int,int,int,int]]):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype=int)
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            input_centroids[i] = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = np.linalg.norm(np.array(object_centroids)[:, None] - input_centroids[None], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows, used_cols = set(), set()
            for r, c in zip(rows, cols):
                if r in used_rows or c in used_cols:
                    continue
                object_id = object_ids[r]
                self.objects[object_id] = tuple(input_centroids[c])
                self.disappeared[object_id] = 0
                used_rows.add(r)
                used_cols.add(c)
            unused_rows = set(range(len(object_centroids))) - used_rows
            unused_cols = set(range(len(input_centroids))) - used_cols
            for r in unused_rows:
                object_id = object_ids[r]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            for c in unused_cols:
                self.register(tuple(input_centroids[c]))

        return self.objects
