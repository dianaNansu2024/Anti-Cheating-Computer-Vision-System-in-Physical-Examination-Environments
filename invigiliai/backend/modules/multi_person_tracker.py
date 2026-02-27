"""
MultiPersonTracker
==================
Detects all people in frame and assigns stable seat-based IDs
using centroid tracking — no face recognition needed.

Each person is tracked across frames by their position (centroid).
IDs are assigned as SEAT-1, SEAT-2, etc. based on horizontal position
(left-to-right), giving stable, meaningful labels.
"""

import cv2
import numpy as np
from collections import OrderedDict


class MultiPersonTracker:
    def __init__(self):
        self.next_id = 1
        self.centroids = OrderedDict()   # id -> centroid (cx, cy)
        self.disappeared = OrderedDict() # id -> frames since last seen
        self.max_disappeared = 30        # frames before removing a track
        self.max_distance = 120          # pixels for centroid matching

        # HOG people detector (built into OpenCV — no model download needed)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Also use a lightweight face detector as fallback for seated students
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def reset(self):
        self.next_id = 1
        self.centroids.clear()
        self.disappeared.clear()

    def detect(self, frame):
        """
        Detect person bounding boxes in frame.
        Returns list of (x, y, w, h) tuples.
        Combines HOG full-body + face detection for better coverage
        of seated students (HOG works best for standing).
        """
        h, w = frame.shape[:2]
        detections = []

        # ── HOG full-body detection ──────────────────────────────────────
        scale = 0.5
        small = cv2.resize(frame, (int(w*scale), int(h*scale)))
        boxes_hog, weights = self.hog.detectMultiScale(
            small,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05,
            finalThreshold=2
        )
        for (x, y, bw, bh) in (boxes_hog if len(boxes_hog) else []):
            detections.append((
                int(x/scale), int(y/scale),
                int(bw/scale), int(bh/scale)
            ))

        # ── Face + upper-body fallback (better for seated) ───────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        for (fx, fy, fw, fh) in (faces if len(faces) else []):
            # Expand face box to approximate upper body
            upper_x = max(0, fx - int(fw * 0.5))
            upper_y = max(0, fy - int(fh * 0.2))
            upper_w = int(fw * 2.0)
            upper_h = int(fh * 3.5)
            detections.append((upper_x, upper_y, upper_w, upper_h))

        # Non-maximum suppression to merge overlapping boxes
        if detections:
            detections = self._nms(detections, overlap_thresh=0.4)

        return detections

    def update(self, frame):
        """
        Run detection + centroid tracking on frame.
        Returns list of { id, bbox, roi, centroid }
        """
        raw_boxes = self.detect(frame)
        h, w = frame.shape[:2]

        if len(raw_boxes) == 0:
            # Mark all existing tracks as disappeared
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.centroids[oid]
                    del self.disappeared[oid]
            return []

        # Compute new centroids
        new_centroids = []
        for (x, y, bw, bh) in raw_boxes:
            cx = x + bw // 2
            cy = y + bh // 2
            new_centroids.append((cx, cy))

        # Register if no existing tracks
        if len(self.centroids) == 0:
            for c in new_centroids:
                self._register(c)
        else:
            # Match new centroids to existing by minimum distance
            existing_ids  = list(self.centroids.keys())
            existing_cents = list(self.centroids.values())

            D = np.linalg.norm(
                np.array(existing_cents)[:, np.newaxis] - np.array(new_centroids),
                axis=2
            )

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                oid = existing_ids[row]
                self.centroids[oid] = new_centroids[col]
                self.disappeared[oid] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for row in unused_rows:
                oid = existing_ids[row]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.centroids[oid]
                    del self.disappeared[oid]

            for col in unused_cols:
                self._register(new_centroids[col])

        # Build output
        results = []
        id_to_box = {}

        for (x, y, bw, bh), (cx, cy) in zip(raw_boxes, new_centroids):
            # Find which tracked ID this centroid maps to
            best_id = None
            best_dist = float("inf")
            for oid, oc in self.centroids.items():
                d = np.linalg.norm(np.array([cx, cy]) - np.array(oc))
                if d < best_dist:
                    best_dist = d
                    best_id = oid

            if best_id is None:
                continue

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            results.append({
                "id":       best_id,
                "bbox":     (x1, y1, x2-x1, y2-y1),
                "roi":      roi,
                "centroid": (cx, cy)
            })

        # Sort by horizontal position (left = SEAT-1)
        results.sort(key=lambda p: p["centroid"][0])
        for i, p in enumerate(results):
            p["seat_label"] = f"{i+1}"

        return results

    def _register(self, centroid):
        self.centroids[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _nms(self, boxes, overlap_thresh=0.4):
        """Simple non-maximum suppression."""
        if not boxes:
            return []
        rects = np.array([[x, y, x+w, y+h] for (x,y,w,h) in boxes], dtype=float)
        x1, y1, x2, y2 = rects[:,0], rects[:,1], rects[:,2], rects[:,3]
        areas = (x2-x1+1) * (y2-y1+1)
        order = areas.argsort()[::-1]
        keep = []

        while len(order):
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2-xx1+1) * np.maximum(0, yy2-yy1+1)
            iou   = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou <= overlap_thresh]

        return [boxes[k] for k in keep]
