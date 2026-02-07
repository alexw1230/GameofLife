import cv2
import numpy as np
from ultralytics import YOLO
import random

# =================== Models ===================
det_model = YOLO("models/yolov8n.pt")        
pose_model = YOLO("models/yolov8n-pose.pt")  

cap = cv2.VideoCapture(0)

# =================== Constants ===================
KNOWN_HEIGHT = 1.7  
FOCAL_LENGTH = 700  
MAX_HP = 100
MAX_MANA = 100
THREAT_THRESHOLD = 120  
PROXIMITY_THRESHOLD = 50  

# =================== Quest System ===================
class QuestManager:
    def __init__(self):
        self.main_quest = "Survive the Simulation"
        self.side_quests = [
            "Scan a high-mana entity",
            "Identify 3 friendly NPCs",
            "Maintain distance from threats",
            "Find the hidden merchant",
            "Analyze biological signatures"
        ]
        self.current_side_quest = random.choice(self.side_quests)
        self.quest_color = (0, 255, 255) # Gold/Yellow

    def refresh_side_quest(self):
        self.current_side_quest = random.choice(self.side_quests)

quest_sys = QuestManager()

# =================== Persistent attributes ===================
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
    max_size_factor = 300  
    hp = int(min(size_factor / max_size_factor * MAX_HP, MAX_HP))
    return hp

def get_persistent_attributes(track_id, bbox, attributes_dict, threshold=PROXIMITY_THRESHOLD):
    if track_id in attributes_dict:
        return (attributes_dict[track_id]['hp'], attributes_dict[track_id]['mana']), track_id
    cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
    for old_id, data in attributes_dict.items():
        ox1, oy1, ox2, oy2 = data['bbox']
        ox, oy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
        if ((cx - ox)**2 + (cy - oy)**2)**0.5 < threshold:
            return (data['hp'], data['mana']), old_id
    return None, track_id

def draw_health_bars(frame, x, y, hp, mana, max_hp=MAX_HP, max_mana=MAX_MANA):
    bar_width, bar_height, spacing = 150, 20, 5
    # Background
    cv2.rectangle(frame, (x - 10, y - 10), (x + 180, y + 60), (20, 20, 20), -1)
    # HP
    hp_fill = int(bar_width * (hp/max_hp))
    cv2.rectangle(frame, (x+5, y+5), (x+5+bar_width, y+25), (50, 50, 100), -1)
    cv2.rectangle(frame, (x+5, y+5), (x+5+hp_fill, y+25), (0, 0, 255), -1)
    # Mana
    mana_fill = int(bar_width * (mana/max_mana))
    cv2.rectangle(frame, (x+5, y+35), (x+5+bar_width, y+55), (100, 50, 50), -1)
    cv2.rectangle(frame, (x+5, y+35), (x+5+mana_fill, y+55), (255, 0, 0), -1)

def draw_quest_ui(frame, quest_manager):
    """Draws a semi-transparent Quest Log in the top right."""
    h, w, _ = frame.shape
    overlay = frame.copy()
    
    # UI Box Dimensions
    box_w, box_h = 300, 120
    tx, ty = w - box_w - 20, 20
    
    # Draw Background Semi-transparent Box
    cv2.rectangle(overlay, (tx, ty), (tx + box_w, ty + box_h), (40, 40, 40), -1)
    cv2.rectangle(overlay, (tx, ty), (tx + box_w, ty + box_h), (200, 200, 200), 2)
    
    # Apply transparency
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Text
    cv2.putText(frame, "QUEST LOG", (tx + 10, ty + 25), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(frame, (tx + 10, ty + 35), (tx + box_w - 10, ty + 35), (150, 150, 150), 1)
    
    # Main Quest
    cv2.putText(frame, f"Main: {quest_manager.main_quest}", (tx + 10, ty + 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    
    # Side Quest
    cv2.putText(frame, f"Side: {quest_manager.current_side_quest}", (tx + 10, ty + 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)

# =================== Fullscreen setup ===================
cv2.namedWindow("YOLOv8 RPG View", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("YOLOv8 RPG View", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# =================== Main loop ===================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = det_model.track(frame, persist=True, classes=[0], conf=0.4, tracker="bytetrack.yaml")

    for r in results:
        if r.boxes.id is None: continue
        for box, track_id in zip(r.boxes, r.boxes.id):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox_height = y2 - y1
            distance = estimate_distance(bbox_height)
            if distance is None: continue

            attr, matched_id = get_persistent_attributes(track_id, (x1, y1, x2, y2), person_attributes)
            if attr:
                hp, mana = attr
            else:
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0: continue
                upper_color = dominant_color(person_crop[:person_crop.shape[0]//2, :])
                hp, mana = size_to_hp(bbox_height, distance), color_to_mana(upper_color)

            person_attributes[matched_id] = {'hp': hp, 'mana': mana, 'bbox': (x1, y1, x2, y2)}
            
            # Threat logic
            box_color = (0, 0, 255) if (bbox_height / distance) > THREAT_THRESHOLD else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            draw_health_bars(frame, (x1 + x2) // 2 - 75, max(10, y1 - 70), hp, mana)

    # Draw the Quest UI
    draw_quest_ui(frame, quest_sys)

    cv2.imshow("YOLOv8 RPG View", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'): # Press 'R' to shuffle quest
        quest_sys.refresh_side_quest()

#deltethis

cap.release()
cv2.destroyAllWindows()