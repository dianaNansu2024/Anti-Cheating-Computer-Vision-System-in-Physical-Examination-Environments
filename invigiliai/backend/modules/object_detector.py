"""
ObjectDetector — Custom Model Edition
=======================================
Loads YOUR trained model (invigilai_best.pt) and automatically:
  - Reads all class names directly from the model
  - Assigns severity (high / medium / low) based on class name keywords
  - Flags ALL classes your model detects (no hardcoded filter list)
  - Falls back gracefully if model file is missing

Drop-in replacement for the generic object_detector.py.
Works for both InvigilAI v2 (live) and v3 (video file).
"""

import os
import cv2
import numpy as np

# ── Confidence threshold ───────────────────────────────────────────────────
# Lower  = more sensitive (more detections, more false positives)
# Higher = more strict   (fewer detections, fewer false alarms)
CONFIDENCE_THRESHOLD = 0.40

# ── Severity keyword rules ─────────────────────────────────────────────────
# Class names containing these keywords get the assigned severity.
# Your model's class names are matched against these (case-insensitive).
# Add your own keywords if your model has custom class names.
SEVERITY_RULES = {
    "high": [
        "phone", "mobile", "cell", "smartphone",
        "earphone", "earpiece", "headphone", "airpod",
        "laptop", "tablet", "ipad", "computer",
        "cheating", "cheat",
        "camera", "smartwatch", "watch",
    ],
    "medium": [
        "book", "paper", "note", "sheet", "card",
        "pencil", "pen", "calculator",
        "whisper", "talking", "mouth",
        "looking", "gaze",
    ],
    "low": [
        "person", "hand", "gesture",
        "normal", "sitting",
    ],
}

# Default severity if no keyword matches
DEFAULT_SEVERITY = "medium"

# Path to your trained model — place invigilai_best.pt in backend/
DEFAULT_MODEL_PATH = "invigilai_best.pt"
FALLBACK_MODEL_PATH = "yolov8n.pt"  # fallback if custom model not found


def assign_severity(class_name: str) -> str:
    """
    Automatically assign a severity level based on the class name.
    Checks keyword lists above — no manual configuration needed.
    """
    name_lower = class_name.lower()
    for severity, keywords in SEVERITY_RULES.items():
        if any(kw in name_lower for kw in keywords):
            return severity
    return DEFAULT_SEVERITY


class ObjectDetector:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.ready      = False
        self.model      = None
        self.class_map  = {}   # class_id -> { name, severity }
        self.model_info = {}   # metadata about loaded model

        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """
        Try to load custom model first, fall back to generic YOLOv8n.
        Auto-discovers all class names and assigns severity to each.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            print("[ObjectDetector] ultralytics not installed. Run: pip install ultralytics")
            return

        # ── Try custom model first ─────────────────────────────────────
        # Check both: passed path and relative to this file's directory
        search_paths = [
            model_path,
            os.path.join(os.path.dirname(__file__), "..", model_path),
            os.path.join(os.path.dirname(__file__), model_path),
        ]

        loaded_path = None
        for path in search_paths:
            path = os.path.normpath(path)
            if os.path.exists(path):
                loaded_path = path
                break

        if loaded_path:
            try:
                self.model = YOLO(loaded_path)
                self._build_class_map()
                self.ready = True
                self.model_info["source"] = "custom"
                self.model_info["path"]   = loaded_path
                print(f"\n{'='*55}")
                print(f"  [ObjectDetector] Custom model loaded!")
                print(f"  Path     : {loaded_path}")
                print(f"  Classes  : {len(self.class_map)}")
                for cid, info in self.class_map.items():
                    print(f"    [{cid}] {info['name']:25s} → {info['severity'].upper()}")
                print(f"{'='*55}\n")
                return
            except Exception as e:
                print(f"[ObjectDetector] Failed to load custom model: {e}")

        # ── Fall back to generic YOLO ──────────────────────────────────
        print(f"[ObjectDetector] Custom model '{model_path}' not found.")
        print(f"[ObjectDetector] Falling back to {FALLBACK_MODEL_PATH}...")
        print(f"[ObjectDetector] To use your model: place invigilai_best.pt in backend/")

        try:
            self.model = YOLO(FALLBACK_MODEL_PATH)
            self._build_class_map(custom=False)
            self.ready = True
            self.model_info["source"] = "fallback"
            self.model_info["path"]   = FALLBACK_MODEL_PATH
            print(f"[ObjectDetector] Fallback loaded. "
                  f"Monitoring {len(self.class_map)} classes.")
        except Exception as e:
            print(f"[ObjectDetector] Fallback also failed: {e}")

    def _build_class_map(self, custom: bool = True):
        """
        Read class names from the loaded model and assign severity to each.
        For the generic COCO model, only map exam-relevant classes.
        """
        self.class_map = {}

        if custom:
            # Custom model: map ALL classes
            for cid, name in self.model.names.items():
                self.class_map[cid] = {
                    "name":     name,
                    "severity": assign_severity(name)
                }
        else:
            # Generic COCO model: only keep exam-relevant classes
            relevant_keywords = (
                list(SEVERITY_RULES["high"]) +
                list(SEVERITY_RULES["medium"])
            )
            for cid, name in self.model.names.items():
                name_lower = name.lower()
                if any(kw in name_lower for kw in relevant_keywords):
                    self.class_map[cid] = {
                        "name":     name,
                        "severity": assign_severity(name)
                    }

    def process(self, roi, offset=(0, 0)):
        """
        Run detection on a person ROI (cropped region of the frame).

        Args:
            roi:    Cropped BGR image of a person region
            offset: (x, y) position of the ROI in the full frame
                    Used to convert ROI bbox coordinates to full-frame coords

        Returns:
            {
              "detected_items": [
                {
                  "label":      str,    class name
                  "confidence": float,  0.0 - 1.0
                  "severity":   str,    high / medium / low
                  "bbox":       tuple,  (x, y, w, h) in full frame coords
                }
              ],
              "model_source":  str     custom / fallback
            }
        """
        output = {
            "detected_items": [],
            "model_source":   self.model_info.get("source", "none")
        }

        if not self.ready or roi is None or roi.size == 0:
            return output

        ox, oy = offset

        try:
            results = self.model(roi, verbose=False)[0]
        except Exception as e:
            print(f"[ObjectDetector] Inference error: {e}")
            return output

        for box in results.boxes:
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])

            # Skip low-confidence detections
            if conf < CONFIDENCE_THRESHOLD:
                continue

            # Skip classes not in our map
            if cls_id not in self.class_map:
                continue

            class_info = self.class_map[cls_id]
            label      = class_info["name"]
            severity   = class_info["severity"]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            output["detected_items"].append({
                "label":      label,
                "confidence": round(conf, 3),
                "severity":   severity,
                "bbox":       (ox + x1, oy + y1, x2 - x1, y2 - y1)
            })

        return output

    def get_model_info(self) -> dict:
        """Return info about the loaded model — useful for dashboard display."""
        return {
            **self.model_info,
            "classes":   {cid: info["name"] for cid, info in self.class_map.items()},
            "n_classes": len(self.class_map),
            "ready":     self.ready,
            "threshold": CONFIDENCE_THRESHOLD,
        }
