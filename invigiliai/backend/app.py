"""
InvigilAI — Unified Edition
=============================
One app. Two modes. Switch at any time.

  LIVE MODE   — Real-time camera feed analysis
  VIDEO MODE  — Upload a pre-recorded video for analysis

Both modes share the same detection pipeline:
  • Custom YOLOv8 model (invigilai_best.pt)
  • Gaze tracking
  • Posture detection
  • Mouth / whispering detection
  • Multi-person tracking + seat IDs
  • Anomaly scoring per seat
"""

import cv2
import time
import os
import threading
from datetime import datetime
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from modules.multi_person_tracker import MultiPersonTracker
from modules.gaze_tracker import GazeTracker
from modules.pose_estimator import PoseEstimator
from modules.object_detector import ObjectDetector
from modules.mouth_monitor import MouthMonitor
from modules.anomaly_scorer import AnomalyScorer
from utils.report_generator import ReportGenerator
from utils.logger import ExamLogger

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTS  = {"mp4", "avi", "mov", "mkv", "webm", "m4v"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ─── Shared State ─────────────────────────────────────────────────────────────
state = {
    # Mode
    "mode":           "none",      # "none" | "live" | "video"

    # Shared
    "status":         "idle",      # idle | ready | active | paused | done | error
    "exam_id":        None,
    "start_time":     None,
    "alerts":         [],
    "persons":        {},
    "frame_count":    0,

    # Live mode
    "camera_index":   0,

    # Video mode
    "video_path":     None,
    "video_name":     None,
    "total_frames":   0,
    "fps":            25.0,
    "duration_sec":   0.0,
    "width":          0,
    "height":         0,
    "current_frame":  0,
    "playback_speed": 1.0,
    "paused":         False,
}

# ─── Module Init ──────────────────────────────────────────────────────────────
tracker        = MultiPersonTracker()
gaze_tracker   = GazeTracker()
pose_estimator = PoseEstimator()
obj_detector   = ObjectDetector("invigilai_best.pt")
mouth_monitor  = MouthMonitor()
scorer         = AnomalyScorer()
logger         = ExamLogger() 
report_gen     = ReportGenerator()

# Shared frame buffer
_frame_lock      = threading.Lock()
_latest_frame    = None
_camera          = None
_video_thread    = None

# ─── Detection Pipeline (shared by both modes) ────────────────────────────────
def process_frame(frame, frame_num=None, fps=None):
    """
    Run full detection pipeline on a single frame.
    frame_num / fps only used in video mode for timestamp annotation.
    """
    incidents = []
    h, w = frame.shape[:2]
    persons = tracker.update(frame)

    if len(persons) == 0:
        incidents.append(_inc("SEAT_VACANT", "high", "No person detected", "?", frame_num, fps))
    if len(persons) > 1:
        incidents.append(_inc("MULTIPLE_PERSONS", "high",
                              f"{len(persons)} people in frame", "?", frame_num, fps))

    for person in persons:
        pid     = person["id"]
        roi     = person["roi"]
        bbox    = person["bbox"]
        signals = {}

        # Gaze
        gaze = gaze_tracker.process(roi, person_id=pid)
        signals["gaze"] = gaze
        if gaze.get("suspicious"):
            incidents.append(_inc("GAZE_DEVIATION", "medium",
                f"Eyes {gaze['direction']} for {gaze.get('duration',0):.1f}s",
                pid, frame_num, fps))

        # Pose
        pose = pose_estimator.process(roi)
        signals["pose"] = pose
        if pose.get("suspicious"):
            incidents.append(_inc("SUSPICIOUS_POSTURE", "medium",
                pose["description"], pid, frame_num, fps))

        # Object detection (custom model)
        objects = obj_detector.process(roi, offset=bbox[:2])
        signals["objects"] = objects
        for obj in objects.get("detected_items", []):
            incidents.append(_inc("UNAUTHORIZED_OBJECT", obj["severity"],
                f"[{obj['severity'].upper()}] {obj['label']} ({obj['confidence']:.0%})",
                pid, frame_num, fps))
            # Draw detection box
            ox, oy, ow, oh = obj["bbox"]
            sc = (0,0,255) if obj["severity"]=="high" else (0,165,255)
            cv2.rectangle(frame, (ox,oy), (ox+ow,oy+oh), sc, 2)
            cv2.putText(frame, f"{obj['label']} {obj['confidence']:.0%}",
                        (ox, oy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, sc, 1)

        # Mouth
        mouth = mouth_monitor.process(roi, person_id=pid)
        signals["mouth"] = mouth
        if mouth.get("talking"):
            incidents.append(_inc("MOUTH_MOVEMENT", "medium",
                f"Talking detected ({mouth.get('duration',0):.1f}s)",
                pid, frame_num, fps))

        # Score
        risk = scorer.calculate(pid, signals)
        state["persons"].setdefault(pid, {"incident_count": 0, "risk": 0})
        state["persons"][pid]["risk"] = risk
        state["persons"][pid]["incident_count"] += len(
            [i for i in incidents if i.get("seat") == pid])

        # Person bounding box annotation
        x, y, bw, bh = bbox
        color = (0,0,255) if risk>60 else (0,165,255) if risk>30 else (0,220,80)
        cv2.rectangle(frame, (x,y), (x+bw,y+bh), color, 2)
        lbl = f"SEAT {pid}  {risk:.0f}%"
        (lw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x, y-22), (x+lw+6, y), (0,0,0), -1)
        cv2.putText(frame, lbl, (x+3, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # ── Info strip ────────────────────────────────────────────────────────
    mode_tag = "LIVE" if state["mode"] == "live" else "VIDEO"
    model_src = obj_detector.model_info.get("source", "?")

    if state["mode"] == "video" and frame_num and fps:
        vt = _fmt_time(frame_num, fps)
        pct = (frame_num / max(state["total_frames"], 1)) * 100
        cv2.rectangle(frame, (0, h-28), (w, h), (0,0,0), -1)
        cv2.putText(frame,
            f"  [{mode_tag}]  {vt}  |  {frame_num}/{state['total_frames']}  "
            f"|  {pct:.1f}%  |  model: {model_src}  |  InvigilAI",
            (8, h-9), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100,200,255), 1)
        bar_w = int(pct / 100 * w)
        cv2.rectangle(frame, (0, h-2), (bar_w, h), (0,200,170), -1)
    else:
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame,
            f"  [{mode_tag}]  {ts}  |  model: {model_src}  |  InvigilAI",
            (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100,200,255), 1)

    if state["status"] == "active":
        cv2.circle(frame, (w-20, 20), 8, (0,0,255), -1)

    if incidents:
        cv2.putText(frame, f"⚠ {len(incidents)}",
                    (w-70, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    return frame, incidents


def _inc(itype, severity, detail, seat, frame_num=None, fps=None):
    entry = {
        "type":      itype,
        "severity":  severity,
        "detail":    detail,
        "seat":      seat,
        "timestamp": datetime.now().isoformat(),
        "mode":      state["mode"],
    }
    if frame_num is not None and fps:
        entry["frame"]      = frame_num
        entry["video_time"] = _fmt_time(frame_num, fps)
    return entry

def _fmt_time(frame_num, fps):
    t = frame_num / max(fps, 1)
    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"

def _commit_incidents(incidents):
    for inc in incidents:
        logger.log_incident(state["exam_id"], inc)
        state["alerts"].insert(0, inc)
    state["alerts"] = state["alerts"][:300]

def _reset_detection():
    scorer.reset_all()
    tracker.reset()
    state.update({
        "alerts":      [],
        "persons":     {},
        "frame_count": 0,
        "current_frame": 0,
    })

# ─── LIVE MODE ────────────────────────────────────────────────────────────────
def _live_stream():
    global _camera, _latest_frame
    while state["mode"] == "live":
        if _camera is None or not _camera.isOpened():
            time.sleep(0.1)
            continue
        ret, frame = _camera.read()
        if not ret:
            time.sleep(0.05)
            continue
        state["frame_count"] += 1
        if state["status"] == "active":
            try:
                annotated, incidents = process_frame(frame)
                _commit_incidents(incidents)
            except Exception as e:
                print(f"[Live] Frame error: {e}")
                annotated = frame
        else:
            annotated = frame
        with _frame_lock:
            _latest_frame = annotated.copy()
        time.sleep(0.033)

# ─── VIDEO MODE ───────────────────────────────────────────────────────────────
def _video_analysis():
    global _latest_frame
    cap = cv2.VideoCapture(state["video_path"])
    if not cap.isOpened():
        state["status"] = "error"
        return

    state["status"]     = "active"
    state["start_time"] = datetime.now().isoformat()
    fps = state["fps"]
    frame_num = 0

    while cap.isOpened():
        while state["paused"] and state["mode"] == "video":
            time.sleep(0.05)
        if state["mode"] != "video":
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        state["current_frame"] = frame_num
        state["frame_count"]   = frame_num

        try:
            annotated, incidents = process_frame(frame, frame_num, fps)
            _commit_incidents(incidents)
        except Exception as e:
            print(f"[Video] Frame {frame_num} error: {e}")
            annotated = frame

        with _frame_lock:
            _latest_frame = annotated.copy()

        delay = 1.0 / (fps * max(state["playback_speed"], 0.1))
        time.sleep(max(delay, 0.01))

    cap.release()
    state["status"] = "done"
    report_gen.generate(state)
    print(f"[Video] Analysis complete — {len(state['alerts'])} incidents.")

# ─── Stream Generator (shared) ────────────────────────────────────────────────
def stream_generator():
    while True:
        with _frame_lock:
            frame = _latest_frame.copy() if _latest_frame is not None else _placeholder()
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 83])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n")
        time.sleep(0.033)

def _placeholder():
    import numpy as np
    img = np.zeros((480, 854, 3), dtype=np.uint8)
    img[:] = (10, 16, 24)
    n_cls = obj_detector.model_info.get("n_classes", 0)
    src   = obj_detector.model_info.get("source", "—")
    cv2.putText(img, "InvigilAI — Unified Edition",
                (230, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,200,170), 2)
    cv2.putText(img, f"model: invigilai_best.pt  [{src}]  |  {n_cls} classes",
                (230, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60,100,130), 1)
    cv2.putText(img, "Select LIVE CAMERA  or  UPLOAD VIDEO  to begin",
                (185, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40,70,90), 1)
    return img

# ─── API — Mode switching ──────────────────────────────────────────────────────
@app.route("/api/set_mode", methods=["POST"])
def set_mode():
    """Switch between live / video mode. Stops current mode cleanly first."""
    global _camera, _video_thread
    data     = request.get_json(silent=True) or {}
    new_mode = data.get("mode")          # "live" or "video"

    if new_mode not in ("live", "video"):
        return jsonify({"error": "mode must be 'live' or 'video'"}), 400

    # ── Tear down current mode ───────────────────────────────────────────
    old_mode = state["mode"]
    state["mode"]   = "none"
    state["status"] = "idle"
    time.sleep(0.1)   # let threads notice mode change

    if _camera:
        _camera.release()
        _camera = None

    _reset_detection()

    # ── Set up new mode ──────────────────────────────────────────────────
    state["mode"]   = new_mode
    state["exam_id"] = f"EXAM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.start_session(state["exam_id"])

    if new_mode == "live":
        cam_idx = data.get("camera_index", 0)
        state["camera_index"] = cam_idx
        _camera = cv2.VideoCapture(cam_idx)
        _camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not _camera.isOpened():
            state["mode"] = "none"
            return jsonify({"error": f"Cannot open camera {cam_idx}"}), 400
        state["status"] = "ready"
        t = threading.Thread(target=_live_stream, daemon=True)
        t.start()
        return jsonify({"status": "ready", "mode": "live", "exam_id": state["exam_id"]})

    else:  # video
        state["status"] = "idle"   # wait for upload
        return jsonify({"status": "idle", "mode": "video", "exam_id": state["exam_id"]})

# ─── API — Live controls ───────────────────────────────────────────────────────
@app.route("/api/live/start", methods=["POST"])
def live_start():
    if state["mode"] != "live":
        return jsonify({"error": "Not in live mode"}), 400
    state["status"]     = "active"
    state["start_time"] = datetime.now().isoformat()
    return jsonify({"status": "active"})

@app.route("/api/live/stop", methods=["POST"])
def live_stop():
    if state["mode"] != "live":
        return jsonify({"error": "Not in live mode"}), 400
    state["status"] = "done"
    path = report_gen.generate(state)
    return jsonify({"status": "done", "report": path})

# ─── API — Video controls ──────────────────────────────────────────────────────
@app.route("/api/video/upload", methods=["POST"])
def video_upload():
    if state["mode"] != "video":
        return jsonify({"error": "Not in video mode"}), 400
    if "video" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["video"]
    if not (f.filename and '.' in f.filename and
            f.filename.rsplit('.',1)[1].lower() in ALLOWED_EXTS):
        return jsonify({"error": "Unsupported format"}), 400

    filename = secure_filename(f.filename)
    path     = os.path.join(UPLOAD_FOLDER, filename)
    f.save(path)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return jsonify({"error": "Cannot read video"}), 400
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    state.update({
        "video_path":   path,
        "video_name":   filename,
        "total_frames": total,
        "fps":          fps,
        "duration_sec": total / max(fps, 1),
        "width":        width,
        "height":       height,
        "status":       "ready",
        "current_frame":0,
    })
    _show_first_frame(path)
    return jsonify({
        "status":       "ready",
        "filename":     filename,
        "fps":          round(fps, 2),
        "frames":       total,
        "duration_fmt": _fmt_time(total, fps),
        "width":        width,
        "height":       height,
    })

def _show_first_frame(path):
    global _latest_frame
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0,0), (w,h), (0,200,170), 3)
        cv2.putText(frame, "READY — Press ANALYZE",
                    (w//2-160, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,170), 2)
        with _frame_lock:
            _latest_frame = frame

@app.route("/api/video/analyze", methods=["POST"])
def video_analyze():
    global _video_thread
    if state["mode"] != "video" or state["status"] != "ready":
        return jsonify({"error": f"Cannot analyze: mode={state['mode']} status={state['status']}"}), 400
    data = request.get_json(silent=True) or {}
    state["playback_speed"] = float(data.get("speed", 1.0))
    state["paused"]         = False
    _reset_detection()
    _video_thread = threading.Thread(target=_video_analysis, daemon=True)
    _video_thread.start()
    return jsonify({"status": "active"})

@app.route("/api/video/pause", methods=["POST"])
def video_pause():
    if state["mode"] == "video" and state["status"] == "active":
        state["paused"] = not state["paused"]
        st = "paused" if state["paused"] else "active"
        state["status"] = st
        return jsonify({"paused": state["paused"], "status": st})
    return jsonify({"error": "Not analyzing"}), 400

@app.route("/api/video/speed", methods=["POST"])
def video_speed():
    data = request.get_json(silent=True) or {}
    state["playback_speed"] = max(0.25, min(float(data.get("speed", 1.0)), 8.0))
    return jsonify({"speed": state["playback_speed"]})

@app.route("/api/video/seek", methods=["POST"])
def video_seek():
    global _latest_frame
    if not state["video_path"]:
        return jsonify({"error": "No video"}), 400
    fn  = int((request.get_json(silent=True) or {}).get("frame", 0))
    cap = cv2.VideoCapture(state["video_path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    cap.release()
    if ret:
        with _frame_lock:
            _latest_frame = frame
    return jsonify({"frame": fn, "time": _fmt_time(fn, state["fps"])})

# ─── API — Shared ──────────────────────────────────────────────────────────────
@app.route("/stream")
def video_stream():
    return Response(stream_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/status")
def get_status():
    pct = ((state["current_frame"] / max(state["total_frames"], 1)) * 100
           if state["mode"] == "video" else 0)
    return jsonify({
        "mode":           state["mode"],
        "status":         state["status"],
        "exam_id":        state["exam_id"],
        "start_time":     state["start_time"],
        "frame_count":    state["frame_count"],
        "current_frame":  state["current_frame"],
        "total_frames":   state["total_frames"],
        "progress_pct":   round(pct, 1),
        "fps":            state["fps"],
        "playback_speed": state["playback_speed"],
        "paused":         state["paused"],
        "alert_count":    len(state["alerts"]),
        "recent_alerts":  state["alerts"][:25],
        "persons":        state["persons"],
        "stats": {
            "high":   sum(1 for a in state["alerts"] if a["severity"] == "high"),
            "medium": sum(1 for a in state["alerts"] if a["severity"] == "medium"),
        },
    })

@app.route("/api/model_info")
def model_info():
    return jsonify(obj_detector.get_model_info())

@app.route("/api/cameras")
def list_cameras():
    available = []
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return jsonify({"cameras": available})

@app.route("/api/reset", methods=["POST"])
def reset():
    global _camera, _latest_frame
    state["mode"]   = "none"
    state["status"] = "idle"
    time.sleep(0.1)
    if _camera:
        _camera.release()
        _camera = None
    _reset_detection()
    with _frame_lock:
        _latest_frame = None
    return jsonify({"status": "reset"})

@app.route("/reports/<path:filename>")
def serve_report(filename):
    return send_from_directory("reports", filename)

@app.route("/")
def index():
    return app.send_static_file("index.html")

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  InvigilAI — Unified Edition")
    print(f"  Model  : invigilai_best.pt [{obj_detector.model_info.get('source','?')}]")
    print(f"  Classes: {obj_detector.model_info.get('n_classes',0)}")
    print("  Modes  : Live Camera  +  Video Upload")
    print("  Open   : http://localhost:5000")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
