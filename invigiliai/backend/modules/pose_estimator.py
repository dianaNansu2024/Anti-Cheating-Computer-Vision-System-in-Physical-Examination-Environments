"""
PoseEstimator — Registration-Free
Detects suspicious body postures using MediaPipe Pose.
Works on cropped person ROI — no identity needed.
"""

import cv2
import mediapipe as mp
import numpy as np


class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=0,              # Fastest model
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )

    def process(self, roi):
        output = {"suspicious": False, "description": ""}

        if roi is None or roi.size == 0:
            return output

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return output

        lm = results.pose_landmarks.landmark
        P  = self.mp_pose.PoseLandmark

        def pt(idx):
            return np.array([lm[idx].x, lm[idx].y])

        try:
            nose        = pt(P.NOSE)
            l_shoulder  = pt(P.LEFT_SHOULDER)
            r_shoulder  = pt(P.RIGHT_SHOULDER)
            l_wrist     = pt(P.LEFT_WRIST)
            r_wrist     = pt(P.RIGHT_WRIST)
            l_hip       = pt(P.LEFT_HIP)
            r_hip       = pt(P.RIGHT_HIP)
        except Exception:
            return output

        flags = []
        mid_shoulder_x = (l_shoulder[0] + r_shoulder[0]) / 2
        mid_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
        hip_y          = (l_hip[1] + r_hip[1]) / 2

        # Excessive lateral lean
        lean = abs(nose[0] - mid_shoulder_x)
        if lean > 0.18:
            side = "left" if nose[0] < mid_shoulder_x else "right"
            flags.append(f"Leaning {side}")

        # Head ducked (under desk)
        if nose[1] > mid_shoulder_y + 0.08:
            flags.append("Head below shoulder level")

        # Wrist below hip (reaching under desk)
        if l_wrist[1] > hip_y + 0.12 or r_wrist[1] > hip_y + 0.12:
            flags.append("Hand(s) below desk level")

        # Body turned sideways
        shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
        if shoulder_width < 0.08:
            flags.append("Body turned sideways")

        if flags:
            output["suspicious"]   = True
            output["description"]  = "; ".join(flags)

        return output
