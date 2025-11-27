import cv2
import config
from yolo_detector import YOLODetector
from centroid_tracker import CentroidTracker
from logger import DetectionLogger

def resize_frame(frame):
    if not config.RESIZE_FRAMES:
        return frame
    return cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

def draw(frame, detections, track_map):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        label = det["label"]
        conf = f"{det['confidence']:.2f}"
        tid = track_map.get(tuple(det["bbox"]))
        text = label
        if config.SHOW_CONFIDENCE:
            text += f" {conf}"
        if config.SHOW_TRACK_ID and tid is not None:
            text += f" ID:{tid}"
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def main():
    detector = YOLODetector()
    tracker = CentroidTracker()
    logger = DetectionLogger()

    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        frame = resize_frame(frame)
        detections = detector.detect(frame)
        rects = [det["bbox"] for det in detections]
        objects = tracker.update(rects)
        track_map = {}
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            min_object = min(objects.keys(), key=lambda k: (objects[k][0] - cx) ** 2 + (objects[k][1] - cy) ** 2)
            track_map[tuple(det["bbox"])] = min_object
            logger.log_detection(det["label"], frame_id, det["confidence"], min_object, det["bbox"])
        draw(frame, detections, track_map)
        cv2.imshow("AI Object Detection & Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
