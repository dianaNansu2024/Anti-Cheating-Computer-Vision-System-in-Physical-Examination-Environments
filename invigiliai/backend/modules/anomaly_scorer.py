"""
AnomalyScorer — Per-person rolling risk score with time decay.
No registration needed. Tracks risk per seat ID.
"""

import time
from collections import deque, defaultdict

WEIGHTS = {
    "gaze_deviation":        12,
    "suspicious_posture":    18,
    "object_high":           55,
    "object_medium":         28,
    "object_low":            10,
    "mouth_movement":        15,
    "multiple_persons":      45,
    "seat_vacant":           20,
}

DECAY_WINDOW = 90   # seconds
MAX_SCORE    = 100


class AnomalyScorer:
    def __init__(self):
        self._logs = defaultdict(deque)  # seat_id -> deque of (timestamp, points)

    def reset_all(self):
        self._logs.clear()

    def calculate(self, seat_id, signals):
        now = time.time()
        self._decay(seat_id, now)
        pts = 0

        gaze = signals.get("gaze", {})
        if gaze.get("suspicious"):
            pts += WEIGHTS["gaze_deviation"]

        pose = signals.get("pose", {})
        if pose.get("suspicious"):
            pts += WEIGHTS["suspicious_posture"]

        for obj in signals.get("objects", {}).get("detected_items", []):
            key = f"object_{obj.get('severity', 'medium')}"
            pts += WEIGHTS.get(key, 10)

        mouth = signals.get("mouth", {})
        if mouth.get("talking"):
            pts += WEIGHTS["mouth_movement"]

        if pts > 0:
            self._logs[seat_id].append((now, pts))

        total = sum(p for _, p in self._logs[seat_id])
        return min(round(total, 1), MAX_SCORE)

    def _decay(self, seat_id, now):
        cutoff = now - DECAY_WINDOW
        log = self._logs[seat_id]
        while log and log[0][0] < cutoff:
            log.popleft()
