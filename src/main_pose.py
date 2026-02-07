import cv2
import numpy as np
from ultralytics import YOLO

# =================== Models ===================
det_model = YOLO("models/yolov8n.pt")        # detection + tracking
pose_model = YOLO("models/yolov8n-pose.pt")  # optional, for pose info

cap = cv2.VideoCapture(0)

# =================== Constants ===================
KNOWN_HEIGHT = 1.7  # meters (average person)
FOCAL_LENGTH = 700  # tune based on your camera
MAX_HP = 100
MAX_MANA = 100
THREAT_THRESHOLD = 120  # size_factor threshold for threat
PROXIMITY_THRESHOLD = 50  # pixels, for matching old track

# =================== Persistent attributes ===================
# track_id -> {'hp': HP, 'mana': Mana, 'bbox': (x1,y1,x2,y2)}
person_attributes = {}

# =================== Helper functions ===================
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

def color_to_mana(rgb):
    r, g, b = rgb
    brightness = (0.299*r + 0.587*g + 0.114*b)
    mana = int((brightness / 255) * MAX_MANA)
    return mana

def size_to_hp(bbox_height, distance):
    if distance is None or bbox_height == 0:
        return 0
    size_factor = bbox_height / distance
    max_size_factor = 300  # adjust for camera/scene
    hp = int(min(size_factor / max_size_factor * MAX_HP, MAX_HP))
    return hp

def get_persistent_attributes(track_id, bbox, attributes_dict, threshold=PROXIMITY_THRESHOLD):
    """
    Returns existing HP/Mana if track_id exists,
    or finds the closest previous bbox within threshold to reuse attributes.
    """
    if track_id in attributes_dict:
        return (attributes_dict[track_id]['hp'], attributes_dict[track_id]['mana']), track_id

    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2

    # Check for previous bbox within threshold
    for old_id, data in attributes_dict.items():
        ox1, oy1, ox2, oy2 = data['bbox']
        ox = (ox1 + ox2) // 2
        oy = (oy1 + oy2) // 2
        distance = ((cx - ox)**2 + (cy - oy)**2)**0.5
        if distance < threshold:
            return (data['hp'], data['mana']), old_id

    return None, track_id

def draw_health_bars(frame, x, y, hp, mana, max_hp=MAX_HP, max_mana=MAX_MANA):
    """Draw RPG-style health and mana bars."""
    bar_width = 150
    bar_height = 20
    spacing = 5
    outline_thickness = 2
    
    # Background panel
    panel_height = bar_height * 2 + spacing * 3
    panel_width = bar_width + 30
    cv2.rectangle(frame, (x - 10, y - 10), (x + panel_width, y + panel_height), (20, 20, 20), -1)
    cv2.rectangle(frame, (x - 10, y - 10), (x + panel_width, y + panel_height), (100, 100, 100), outline_thickness)
    
    # HP Bar
    hp_ratio = max(0, min(hp / max_hp, 1.0))
    hp_bar_width = int(bar_width * hp_ratio)
    
    # HP background (dark red)
    cv2.rectangle(frame, (x + 5, y + 5), (x + bar_width + 5, y + bar_height + 5), (50, 50, 100), -1)
    # HP fill (bright red)
    cv2.rectangle(frame, (x + 5, y + 5), (x + 5 + hp_bar_width, y + bar_height + 5), (0, 0, 255), -1)
    # HP border
    cv2.rectangle(frame, (x + 5, y + 5), (x + bar_width + 5, y + bar_height + 5), (150, 150, 150), outline_thickness)
    
    # HP text
    cv2.putText(frame, f"HP {hp}/{max_hp}", (x + 10, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Mana Bar
    mana_ratio = max(0, min(mana / max_mana, 1.0))
    mana_bar_width = int(bar_width * mana_ratio)
    
    # Mana background (dark blue)
    cv2.rectangle(frame, (x + 5, y + bar_height + spacing + 5), 
                  (x + bar_width + 5, y + bar_height * 2 + spacing + 5), (100, 50, 50), -1)
    # Mana fill (bright blue)
    cv2.rectangle(frame, (x + 5, y + bar_height + spacing + 5), 
                  (x + 5 + mana_bar_width, y + bar_height * 2 + spacing + 5), (255, 0, 0), -1)
    # Mana border
    cv2.rectangle(frame, (x + 5, y + bar_height + spacing + 5), 
                  (x + bar_width + 5, y + bar_height * 2 + spacing + 5), (150, 150, 150), outline_thickness)
    
    # Mana text
    cv2.putText(frame, f"Mana {mana}/{max_mana}", (x + 10, y + bar_height * 2 + spacing + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

# =================== Fullscreen setup ===================
cv2.namedWindow("YOLOv8 RPG View", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("YOLOv8 RPG View", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# =================== Main loop ===================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection + tracking
    results = det_model.track(
        frame,
        persist=True,
        classes=[0],  # only people
        conf=0.4,
        tracker="bytetrack.yaml"
    )

    for r in results:
        if r.boxes.id is None:
            continue

        for box, track_id in zip(r.boxes, r.boxes.id):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox_height = y2 - y1
            distance = estimate_distance(bbox_height)
            if distance is None:
                continue

            # =================== Persistent HP/Mana ===================
            attr, matched_id = get_persistent_attributes(track_id, (x1, y1, x2, y2), person_attributes)
            if attr:
                hp, mana = attr
            else:
                # First time seeing this person
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue
                h = person_crop.shape[0]
                upper = person_crop[:h//2, :]
                upper_color = dominant_color(upper)

                hp = size_to_hp(bbox_height, distance)
                mana = color_to_mana(upper_color)

            # Update dictionary with current bbox
            person_attributes[matched_id] = {'hp': hp, 'mana': mana, 'bbox': (x1, y1, x2, y2)}

            # =================== Threat logic ===================
            size_factor = bbox_height / distance
            threat = size_factor > THREAT_THRESHOLD
            box_color = (0, 0, 255) if threat else (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Draw RPG-style health and mana bars above the person
            bar_x = (x1 + x2) // 2 - 75  # Center horizontally (bar_width/2 = 75)
            bar_y = max(10, y1 - 70)  # Above the person, with safety margin
            draw_health_bars(frame, bar_x, bar_y, hp, mana)

    cv2.imshow("YOLOv8 RPG View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
