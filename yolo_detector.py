from ultralytics import YOLO
from typing import List, Dict
import numpy as np
import config

class YOLODetector:
    def __init__(self, model_path: str | None = None, conf_threshold: float | None = None):
        self.model = YOLO(model_path or config.YOLO_MODEL_PATH)
        self.conf_threshold = conf_threshold if conf_threshold is not None else config.CONFIDENCE_THRESHOLD

    def detect(self, frame: np.ndarray) -> List[Dict]:
        results = self.model(frame, verbose=False)[0]
        detections = []
        if results.boxes is None:
            return detections
        for box in results.boxes:
            conf = float(box.conf.item())
            if conf < self.conf_threshold:
                continue
            label = self.model.names[int(box.cls.item())]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({"label": label, "confidence": conf, "bbox": (int(x1), int(y1), int(x2), int(y2))})
        return detections
