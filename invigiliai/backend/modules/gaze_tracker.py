"""
GazeTracker — Registration-Free
Tracks eye gaze direction using MediaPipe Face Mesh iris landmarks.
Works on any person's face without prior enrollment.
"""

import cv2
import mediapipe as mp
import numpy as np
import time

LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
GAZE_THRESHOLD    = 0.30   # Iris displacement ratio to flag
SUSTAINED_SECONDS = 2.0    # Seconds before flagging


class GazeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )
        # Per-person deviation tracking (keyed by caller)
        self._state = {}

    def process(self, roi, person_id=None):
        """
        Analyze gaze from a cropped person ROI.
        Returns: { direction, suspicious, duration }
        """
        key = person_id or "default"
        if key not in self._state:
            self._state[key] = {"start": None, "direction": "center"}

        output = {"direction": "center", "suspicious": False, "duration": 0.0}

        if roi is None or roi.size == 0:
            return output

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return output

        lm = results.multi_face_landmarks[0].landmark
        h, w = roi.shape[:2]

        def pt(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        left_iris  = np.mean([pt(i) for i in LEFT_IRIS],  axis=0)
        right_iris = np.mean([pt(i) for i in RIGHT_IRIS], axis=0)

        # Eye corners
        ll = pt(33);  lr = pt(133)
        rl = pt(362); rr = pt(263)

        def ratio(iris, eye_l, eye_r):
            ew = np.linalg.norm(eye_r - eye_l) + 1e-6
            disp = np.dot(iris - eye_l, eye_r - eye_l) / ew
            return disp / ew

        lr_val = (ratio(left_iris, ll, lr) + ratio(right_iris, rl, rr)) / 2

        if lr_val < 0.5 - GAZE_THRESHOLD:
            direction = "LEFT"
        elif lr_val > 0.5 + GAZE_THRESHOLD:
            direction = "RIGHT"
        else:
            direction = "center"

        # Vertical: head tilt down (under desk)
        nose_y = lm[1].y
        chin_y = lm[152].y
        if chin_y > 0 and (nose_y / chin_y) < 0.36:
            direction = "DOWN"

        output["direction"] = direction

        state = self._state[key]
        if direction != "center":
            if state["start"] is None or state["direction"] != direction:
                state["start"]     = time.time()
                state["direction"] = direction
            else:
                dur = time.time() - state["start"]
                output["duration"] = dur
                if dur >= SUSTAINED_SECONDS:
                    output["suspicious"] = True
        else:
            state["start"]     = None
            state["direction"] = "center"

        return output
