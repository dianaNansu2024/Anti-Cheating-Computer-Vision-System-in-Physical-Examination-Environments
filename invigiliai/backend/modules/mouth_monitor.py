"""
MouthMonitor
============
Detects sustained mouth movement (talking / whispering) using
MediaPipe Face Mesh lip landmarks — no registration needed.
"""

import cv2
import mediapipe as mp
import numpy as np
import time

# Lip landmark indices (upper and lower inner lip)
UPPER_LIP = [13, 312, 311, 310, 415, 308]
LOWER_LIP = [14,  82,  81,  80,  88,  95]

OPEN_THRESHOLD     = 0.04   # Lip separation ratio to consider mouth "open"
TALKING_DURATION   = 1.5    # Seconds of sustained movement before flagging


class MouthMonitor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )
        self._state = {}   # person_id -> state

    def process(self, roi, person_id=None):
        """
        Returns: { talking: bool, duration: float, ratio: float }
        """
        key = person_id or "default"
        if key not in self._state:
            self._state[key] = {"open_since": None, "open": False}

        output = {"talking": False, "duration": 0.0, "ratio": 0.0}

        if roi is None or roi.size == 0:
            return output

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return output

        lm = results.multi_face_landmarks[0].landmark
        h, w = roi.shape[:2]

        def lm_pt(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        # Mouth aspect ratio: vertical opening / horizontal width
        upper_center = np.mean([lm_pt(i) for i in UPPER_LIP], axis=0)
        lower_center = np.mean([lm_pt(i) for i in LOWER_LIP], axis=0)
        mouth_open = np.linalg.norm(upper_center - lower_center)

        # Mouth width (corner to corner)
        left_corner  = lm_pt(61)
        right_corner = lm_pt(291)
        mouth_width  = np.linalg.norm(right_corner - left_corner) + 1e-6

        ratio = mouth_open / mouth_width
        output["ratio"] = round(ratio, 3)

        state = self._state[key]
        if ratio > OPEN_THRESHOLD:
            if state["open_since"] is None:
                state["open_since"] = time.time()
            dur = time.time() - state["open_since"]
            output["duration"] = dur
            if dur >= TALKING_DURATION:
                output["talking"] = True
        else:
            state["open_since"] = None

        return output
