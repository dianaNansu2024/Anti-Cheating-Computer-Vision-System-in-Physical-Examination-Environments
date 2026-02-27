import json, os
from datetime import datetime

class ExamLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.current = None
        os.makedirs(log_dir, exist_ok=True)

    def start_session(self, exam_id):
        self.current = f"{self.log_dir}/{exam_id}.json"
        with open(self.current, "w") as f:
            json.dump({"exam_id": exam_id, "started": datetime.now().isoformat(), "incidents": []}, f)

    def log_incident(self, exam_id, incident):
        if not self.current or not os.path.exists(self.current):
            return
        with open(self.current, "r+") as f:
            data = json.load(f)
            data["incidents"].append(incident)
            f.seek(0); json.dump(data, f, indent=2); f.truncate()
