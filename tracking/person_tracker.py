import time
import numpy as np
from collections import OrderedDict

class PersonTracker:
    def __init__(self, max_disappeared=40, max_trajectory_points=50):
        self.next_object_id = 1
        self.objects = OrderedDict() # id -> centroid (cx, cy)
        self.bboxes = OrderedDict()  # id -> current bbox (x1, y1, x2, y2)
        self.disappeared = OrderedDict() # id -> count of frames disappeared
        
        # Trajectories and Metadata
        self.trajectories = OrderedDict() # id -> list of (cx, cy)
        self.first_seen = OrderedDict()   # id -> timestamp when first registered
        self.last_seen = OrderedDict()    # id -> timestamp when last updated
        self.zones_visited = OrderedDict() # id -> set of zone names visited
        
        self.max_disappeared = max_disappeared
        self.max_trajectory_points = max_trajectory_points

    def register(self, centroid, bbox):
        object_id = self.next_object_id
        self.objects[object_id] = centroid
        self.bboxes[object_id] = bbox
        self.disappeared[object_id] = 0
        px1, py1, px2, py2 = bbox
        bottom_center = (int((px1 + px2) / 2.0), py2)
        self.trajectories[object_id] = [bottom_center]
        self.first_seen[object_id] = time.time()
        self.last_seen[object_id] = time.time()
        self.zones_visited[object_id] = set()
        
        self.next_object_id += 1
        return object_id

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.bboxes:
            del self.bboxes[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
        # Keep trajectory and timing around for a bit or let analyzer handle it before deleting
        # In this tracker, we can clean them up or let the app read them first.
        # Let's delete them to prevent memory leak, but we could also archive them if needed.
        # Let's just delete them now, since events are saved to the database.
        if object_id in self.trajectories:
            del self.trajectories[object_id]
        if object_id in self.first_seen:
            del self.first_seen[object_id]
        if object_id in self.last_seen:
            del self.last_seen[object_id]
        if object_id in self.zones_visited:
            del self.zones_visited[object_id]

    def update(self, rects):
        # rects: list of bounding boxes (x1, y1, x2, y2) representing detected people
        if len(rects) == 0:
            # Increment disappeared count for all existing objects
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.bboxes

        # Calculate centroids of incoming rectangles
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)

        # If we are not tracking anything, register all input centroids
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i], rects[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Calculate distances between existing centroids and input centroids
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

            # Find matching centroids (minimum distances)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                # Threshold to prevent matching centroids that are too far apart (e.g. 150px)
                # If distance is too high, do not match them, let it be a new object
                if D[row, col] > 200:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.bboxes[object_id] = rects[col]
                self.disappeared[object_id] = 0
                self.last_seen[object_id] = time.time()
                
                # Append bottom-center coordinate (cx, y2) to trajectory
                px1, py1, px2, py2 = rects[col]
                bottom_center = (int((px1 + px2) / 2.0), py2)
                self.trajectories[object_id].append(bottom_center)
                if len(self.trajectories[object_id]) > self.max_trajectory_points:
                    self.trajectories[object_id].pop(0)

                used_rows.add(row)
                used_cols.add(col)

            # Check for unmatched rows (existing objects that disappeared)
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Check for unmatched columns (new objects detected)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col], rects[col])

        return self.bboxes

    def get_dwell_time(self, object_id):
        if object_id in self.first_seen:
            return time.time() - self.first_seen[object_id]
        return 0.0

    def get_trajectory(self, object_id):
        return self.trajectories.get(object_id, [])

    def record_zone_visit(self, object_id, zone_name):
        if object_id in self.zones_visited:
            self.zones_visited[object_id].add(zone_name)
