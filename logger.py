import os
import pandas as pd
from datetime import datetime
import config

class DetectionLogger:
    def __init__(self, csv_path: str | None = None):
        self.csv_path = csv_path or config.DETECTION_LOG_PATH
        if not os.path.exists(self.csv_path):
            pd.DataFrame(columns=["timestamp", "frame_id", "label", "confidence", "track_id", "bbox"]).to_csv(self.csv_path, index=False)

    def log_detection(self, label, frame_id, confidence, track_id, bbox):
        df = pd.DataFrame([{
            "timestamp": datetime.now().isoformat(),
            "frame_id": frame_id,
            "label": label,
            "confidence": confidence,
            "track_id": track_id,
            "bbox": bbox
        }])
        df.to_csv(self.csv_path, mode="a", header=False, index=False)
