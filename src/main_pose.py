import cv2
import numpy as np
from ultralytics import YOLO

# Models
det_model = YOLO("models/yolov8n.pt")        # detection + tracking
pose_model = YOLO("models/yolov8n-pose.pt")  # pose estimation

cap = cv2.VideoCapture(0)
cv2.namedWindow("YOLOv8 Threat Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("YOLOv8 Threat Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# Distance calibration constant (tune this!)
KNOWN_HEIGHT = 1.7   # meters (average human)
FOCAL_LENGTH = 700   # arbitrary constant

def estimate_distance(bbox_height_px):
    if bbox_height_px == 0:
        return None
    return (KNOWN_HEIGHT * FOCAL_LENGTH) / bbox_height_px

def dominant_color(img):
    img = img.reshape((-1, 3))
    img = np.float32(img)
    _, _, centers = cv2.kmeans(
        img, 1, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS
    )
    return tuple(map(int, centers[0]))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Track people
    results = det_model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.4,
        tracker="bytetrack.yaml"
    )

    pose_results = pose_model(frame, conf=0.4)

    for r in results:
        if r.boxes.id is None:
            continue

        for box, track_id in zip(r.boxes, r.boxes.id):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox_height = y2 - y1
            distance = estimate_distance(bbox_height)

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            # Split upper / lower body
            h = person_crop.shape[0]
            upper = person_crop[:h//2, :]
            lower = person_crop[h//2:, :]

            upper_color = dominant_color(upper)
            lower_color = dominant_color(lower)

            # Pose heuristic
            aspect_ratio = bbox_height / (x2 - x1 + 1)
            pose = "Standing" if aspect_ratio > 1.6 else "Sitting"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Attribute text
            attributes = [
                f"ID: {int(track_id)}",
                f"BBox H(px): {bbox_height}",
                f"Dist(m): {distance:.2f}" if distance else "Dist: N/A",
                f"Upper color: {upper_color}",
                f"Lower color: {lower_color}",
                f"Pose: {pose}"
            ]

            for i, text in enumerate(attributes):
                cv2.putText(
                    frame,
                    text,
                    (x2 + 10, y1 + 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

    cv2.imshow("YOLOv8 Threat Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
