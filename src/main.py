import argparse
import textwrap
import os
import random
import time
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from pathlib import Path
from openai import OpenAI
from quester import check_quest_reminder

try:
    import yaml
except Exception:
    yaml = None

# Load OpenAI API key from environment or key.env
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # Try to load from key.env in the project root
    env_path = os.path.join(os.path.dirname(__file__), '..', 'key.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip().startswith('api_key'):
                    value = line.split('=', 1)[1].strip().strip('"').strip("'")
                    OPENAI_API_KEY = value
                    break
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable not set and key.env not found or missing api_key entry.")
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.featherless.ai/v1"
)

def generate_mainquest():
   q = check_quest_reminder()
   print(q)
   prompt = f"Here is a task I have to do: {q}. Please make it sound like a medieval quest. No more than 5 words, hard limit. Don't actually use the word quest in the response. The receiver should easily be able to recognize what the task is."
   print(prompt)
   response = client.chat.completions.create(
   model="meta-llama/Meta-Llama-3.1-8B-Instruct",
   messages=[
   {"role": "user", "content": prompt}
   ]
   )
   return q, response.choices[0].message.content

def encode_image_to_base64(image_path):
    """Encode a local image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
def check_quest_complete(image_path,quest):
    base64_image = encode_image_to_base64(image_path)

    # Determine the image format for the data URL
    image_extension = Path(image_path).suffix.lower()
    if image_extension == '.png':
        data_url = f"data:image/png;base64,{base64_image}"
    elif image_extension in ['.jpg', '.jpeg']:
        data_url = f"data:image/jpeg;base64,{base64_image}"
    elif image_extension == '.webp':
        data_url = f"data:image/webp;base64,{base64_image}"
    else:
        data_url = f"data:image/jpeg;base64,{base64_image}"# Default to JPEG
    response = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"The image uploaded is meant to be proof the user completed the following task: {quest}. Please return True if the evidence is sufficent and False otherwise. You should ONLY return the word true or false, nothing more"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]
            }
        ]
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content == "true"



def split_into_three(s):
    parts = s.split(",", 2)  # split on the first two commas only
    if len(parts) < 3:
        raise ValueError("Input string must contain at least two commas")
    return parts[0], parts[1], parts[2]
def resp(title,hp,mana):
    prompt = f"Write a short 2-3 sentence description in the style of a fantasy rpg game. Here is the information you know about them: they are a {title}, hp={hp}, mana={mana}. Don’t use a name or a gender. Make the description expand on their title, even if you have to make stuff up for them. This short description will be used on the bottom of a character card. Also write a more detailed title for them. Your response should read: “{title}, Title, Description” separated by commas. Do not put anything else."
    response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
    {"role": "user", "content": prompt}
    ]
    )
    return response.choices[0].message.content
#Example use (title, hp, mana)
#title, long_title, desc = split_into_three(resp("Mage", 35, 95))



# =================== Constants ===================
KNOWN_HEIGHT = 1.7  # meters (average person)
FOCAL_LENGTH = 400  # tune based on your camera
MAX_HP = 100
MAX_MANA = 100
THREAT_THRESHOLD = 0.6  # ratio threshold (upper_height / expected_px_at_ref_distance)
PROXIMITY_THRESHOLD = 50  # pixels, for matching old track
SMOOTHING_ALPHA = 0.6  # EMA weight for new measurements (0..1)

# HSV color ranges for clothing classification
HSV_COLOR_RANGES = {
    'red': {
        'lower1': np.array([0, 100, 100]),
        'upper1': np.array([10, 255, 255]),
        'lower2': np.array([170, 100, 100]),
        'upper2': np.array([180, 255, 255])
    },
    'blue': {
        'lower': np.array([100, 100, 100]),
        'upper': np.array([130, 255, 255])
    },
    'green': {
        'lower': np.array([40, 100, 100]),
        'upper': np.array([80, 255, 255])
    },
    'yellow': {
        'lower': np.array([20, 100, 100]),
        'upper': np.array([40, 255, 255])
    },
    'purple': {
        'lower': np.array([130, 100, 100]),
        'upper': np.array([160, 255, 255])
    },
    'black': {
        'lower': np.array([0, 0, 0]),
        'upper': np.array([180, 100, 50])
    }
}

# Job colors for visualization
JOB_COLORS = {
    'Tank': (255, 0, 0),          # Blue
    'Warrior': (0, 0, 255),        # Red
    'Warlock': (255, 0, 255),      # Magenta
    'Mage': (255, 255, 0),         # Cyan
    'Healer': (0, 255, 0),         # Green
    'Muggle': (128, 128, 128),     # Gray
    'Commoner': (165, 42, 42),     # Brown
    'Blacksmith': (0, 165, 255),   # Orange
    'Noble': (0, 215, 255),        # Gold
    'Baker': (180, 105, 105),      # Tan
    'Farmer': (34, 139, 34),       # Dark Green
}

# =================== MOUSE INTERACTION GLOBALS ===================
# We need these so the mouse listener knows what's happening in the main loop
latest_frame = None
click_regions = []  # Will store tuples: (x1, y1, x2, y2, track_id)
# =================== Quest Log Click/Result Globals ===================
# For clickable quest log and result overlay
quest_log_regions = []  # (x1, y1, x2, y2, 'main'/'side', quest_text)
quest_result_overlay = {'show': False, 'success': None, 'quest': '', 'timestamp': 0}
# =================== Persistent attributes ===================
# track_id -> {'hp': HP, 'mana': Mana, 'job': job, 'bbox': (x1,y1,x2,y2), 'upper_bbox': (...), ...}
person_attributes = {}

def draw_text_box(img, text_list, x, y, font_scale=0.6, color=(255, 255, 255), thickness=1):
    """Helper to draw multiple lines of text."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 25
    for i, line in enumerate(text_list):
        cv2.putText(img, line.strip(), (x, y + (i * line_height)), font, font_scale, color, thickness)

def mouse_callback(event, x, y, flags, param):
    """Handles mouse clicks, calls AI, and shows Character Card."""
    global latest_frame, click_regions, person_attributes, quest_log_regions, quest_result_overlay
    
    # Trigger only on Left Mouse Button Click (LBUTTONDOWN)
    if event == cv2.EVENT_LBUTTONDOWN:
        # 1. Check if click is in quest log region
        for (qx1, qy1, qx2, qy2, quest_type, quest_text) in quest_log_regions:
            if qx1 < x < qx2 and qy1 < y < qy2:
                print(f"[DEBUG] Clicked quest box: {quest_type} quest '{quest_text}' at ({x},{y})")
                # Take screenshot and check quest completion
                if latest_frame is not None:
                    import time
                    screenshot_path = f"quest_submit_{quest_type}_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_path, latest_frame)
                    # Call check_quest_complete
                    try:
                        if quest_type == 'main':
                            q = check_quest_reminder()
                            result = check_quest_complete(screenshot_path, q)
                        else:
                            result = check_quest_complete(screenshot_path, quest_text)
                    except Exception as e:
                        print(f"Quest check error: {e}")
                        result = False
                    # Delete the screenshot after checking
                    try:
                        if os.path.exists(screenshot_path):
                            os.remove(screenshot_path)
                    except Exception as e:
                        print(f"Error deleting screenshot: {e}")
                    quest_result_overlay['show'] = True
                    quest_result_overlay['success'] = result
                    quest_result_overlay['quest'] = quest_text
                    quest_result_overlay['timestamp'] = time.time()
                return

        # 2. Check if the click (x, y) is inside any of the detected boxes (character card)
        for (x1, y1, x2, y2, track_id) in click_regions:
            if x1 < x < x2 and y1 < y < y2:
                print(f"✅ Clicked on Person ID: {track_id}. Generating AI Description")
                # --- STEP 1: USE STORED USER DATA ---
                data = person_attributes.get(track_id)
                if data is None: return
                job = data['job']
                hp = data['hp']
                mana = data['mana']
                # --- STEP 2: CALL AI ---
                try:
                    print(f"   [Stats] Job: {job} | HP: {hp} | MP: {mana}") # Debug print
                    ai_response = resp(job, hp, mana)
                    title, unique_title, desc = split_into_three(ai_response)
                except Exception as e:
                    print(f"AI Error: {e}")
                    title, unique_title, desc = (str(job), "The Unknown", "AI generation failed.")
                # 3. Prepare the Image (Create Card)
                if latest_frame is not None:
                    h_img, w_img, _ = latest_frame.shape
                    cx1, cy1 = max(0, x1), max(0, y1)
                    cx2, cy2 = min(w_img, x2), min(h_img, y2)
                    if cx2 <= cx1 or cy2 <= cy1:
                        return
                    person_crop = latest_frame[cy1:cy2, cx1:cx2].copy()
                    if person_crop.size > 0:
                        card_width = 900
                        card_height = 800
                        text_area_height = 200
                        img_area_height = card_height - text_area_height
                        img_h, img_w = person_crop.shape[:2]
                        scale = min(card_width / img_w, img_area_height / img_h)
                        new_w = int(img_w * scale)
                        new_h = int(img_h * scale)
                        card_img = cv2.resize(person_crop, (new_w, new_h))
                        card = np.zeros((card_height, card_width, 3), dtype=np.uint8)
                        x_offset = (card_width - new_w) // 2
                        y_offset = (img_area_height - new_h) // 2
                        card[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = card_img
                        text_y = img_area_height + 40
                        cv2.putText(card, unique_title.strip(), (20, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 215, 255), 2)
                        cv2.putText(card, title.strip(), (20, text_y + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 1)
                        desc_lines = textwrap.wrap(desc, width=80)  # Adjust width for card
                        max_lines = 5  # Limit lines to fit in text area
                        desc_lines = desc_lines[:max_lines]
                        draw_text_box(card, desc_lines, 20, text_y + 80)
                        window_name = f"Card: {track_id}"
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(window_name, card_width, card_height)
                        cv2.imshow(window_name, card)
                        cv2.waitKey(1)
                        print(f"   -> Opened Card for ID: {track_id}")
                        break  # Stop after showing the card for the first valid match
                return # Stop checking after finding the first match


# =================== Helper functions ===================
def estimate_distance(bbox_height_px: int) -> float | None:
    """Estimate distance (meters) from pixel height using a pinhole camera model."""
    if bbox_height_px <= 0:
        return None
    return (KNOWN_HEIGHT * FOCAL_LENGTH) / float(bbox_height_px)


def dominant_color(img: np.ndarray) -> tuple[int, int, int]:
    """Return the dominant color (BGR) of an image region using k-means (k=1)."""
    if img is None or img.size == 0:
        return (0, 0, 0)

    data = img.reshape((-1, 3)).astype(np.float32)
    # kmeans to find dominant color
    _, _, centers = cv2.kmeans(
        data, 1, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS,
    )
    return tuple(map(int, centers[0]))


def assign_job(hp: int, mana: int) -> str:
    """Assign a job based on HP and Mana requirements.

    If multiple requirement-based jobs match, randomly pick one of them.
    There's a 15% chance to instead drop to a random fallback job even when
    requirement-matching jobs exist.

    Requirement-based jobs:
    - Tank: HP > 80
    - Warlock: HP > 65 and Mana > 60
    - Warrior: HP > 65 and Mana > 30
    - Mage: HP < 30 and Mana > 75
    - Healer: HP < 30 and Mana > 75 (same requirements as Mage)
    - Muggle: Mana < 10

    Fallback jobs (random): Commoner, Blacksmith, Noble, Baker, Farmer
    """
    fallbacks = ['Commoner', 'Blacksmith', 'Noble', 'Baker', 'Farmer']

    matches: list[str] = []
    if hp >= 65:
        matches.append('Tank')
    if hp >= 50 and mana >= 50:
        matches.append('Warlock')
    if hp >= 50 and mana >= 30:
        matches.append('Warrior')
    if hp <= 30 and mana >= 50:
        # Both Mage and Healer share these requirements
        matches.append('Mage')
        matches.append('Healer')
    if mana <= 10:
        matches.append('Muggle')

    # If there are matches, choose among them, but with 15% chance pick a fallback
    if matches:
        if random.random() < 0.15:
            return random.choice(fallbacks)
        return random.choice(matches)

    # No requirement matches: pick a fallback job
    return random.choice(fallbacks)



def color_to_mana(rgb: tuple[int, int, int]) -> int:
    """Map an RGB/BGR color to a mana value (simple brightness -> mana)."""
    r, g, b = rgb
    brightness = (0.299 * r + 0.587 * g + 0.114 * b)
    mana = int((brightness / 255.0) * MAX_MANA)
    return max(0, min(mana, MAX_MANA))


def size_to_hp(bbox_height: int, distance: float, ref_distance: float = 2.0, scale: float = 1.0) -> int:
    """Compute HP from bounding-box height using a reference-distance normalization.

    Args:
        bbox_height: observed pixel height (upper-body)
        distance: estimated distance in meters (unused for default method but kept for API)
        ref_distance: reference distance in meters used for normalization
        scale: multiplicative scale to calibrate HP output (tunable via config)

    Returns:
        int HP value between 0 and MAX_HP
    """
    if distance is None or bbox_height <= 0:
        return 0

    # Expected pixel height of a person at the reference distance
    expected_px = (KNOWN_HEIGHT * FOCAL_LENGTH) / ref_distance
    if expected_px <= 0:
        return 0

    ratio = float(bbox_height) / float(expected_px)
    hp = int(min(max(ratio * MAX_HP * float(scale), 0), MAX_HP))
    return hp


def expected_pixel_height(distance: float) -> float:
    """Return expected pixel height for a person at given distance (meters)."""
    if distance is None or distance <= 0:
        return 0.0
    return (KNOWN_HEIGHT * FOCAL_LENGTH) / float(distance)


def get_persistent_attributes(track_id, bbox, attributes_dict, threshold=PROXIMITY_THRESHOLD):
    """Return existing (hp,mana) for a track or match a previous entry.

    Matching strategy:
    1. If `track_id` exists, return it.
    2. Try IoU-based matching with previous bboxes (preferred).
    3. Fall back to proximity (centroid distance) matching within `threshold`.
    """
    def bbox_iou(a, b):
        # a and b are (x1,y1,x2,y2)
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return (inter / union) if union > 0 else 0.0

    # Exact track id match
    if track_id in attributes_dict:
        data = attributes_dict[track_id]
        return (data['hp'], data['mana']), track_id

    # IoU matching (preferred) to handle re-detections when track ids change
    best_iou = 0.0
    best_id = None
    for old_id, data in attributes_dict.items():
        prev_bbox = data.get('bbox')
        if not prev_bbox:
            continue
        iou = bbox_iou(prev_bbox, bbox)
        if iou > best_iou:
            best_iou = iou
            best_id = old_id

    # If IoU is sufficiently high, consider it the same person
    if best_iou > 0.3 and best_id is not None:
        data = attributes_dict[best_id]
        return (data['hp'], data['mana']), best_id

    # Fallback: proximity (centroid distance)
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    for old_id, data in attributes_dict.items():
        ox1, oy1, ox2, oy2 = data.get('bbox', (0, 0, 0, 0))
        ox = (ox1 + ox2) // 2
        oy = (oy1 + oy2) // 2
        dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
        if dist < threshold:
            return (data['hp'], data['mana']), old_id

    return None, track_id


def draw_health_bars(frame: np.ndarray, x: int, y: int, hp: int, mana: int, max_hp=MAX_HP, max_mana=MAX_MANA, scale: float = 1.0):
    """Draw RPG-style health and mana bars at (x,y) top-left of panel, scaled by `scale`.

    `scale` is clamped so bars remain readable. Use `scale` in (0.5..1.0].
    """
    # Base sizes (when scale == 1.0)
    base_bar_width = 150
    base_bar_height = 20
    spacing = 5
    outline_thickness = max(1, int(2 * scale))

    # Apply scale but ensure minimum sizes
    bar_width = max(40, int(base_bar_width * scale))
    bar_height = max(8, int(base_bar_height * scale))

    panel_height = bar_height * 2 + spacing * 3
    panel_width = bar_width + 30

    # Panel background and border
    cv2.rectangle(frame, (x - 10, y - 10), (x + panel_width, y + panel_height), (20, 20, 20), -1)
    cv2.rectangle(frame, (x - 10, y - 10), (x + panel_width, y + panel_height), (100, 100, 100), outline_thickness)

    # HP
    hp_ratio = max(0.0, min(float(hp) / float(max_hp), 1.0))
    hp_bar_width = int(bar_width * hp_ratio)
    cv2.rectangle(frame, (x + 5, y + 5), (x + bar_width + 5, y + bar_height + 5), (50, 50, 100), -1)
    cv2.rectangle(frame, (x + 5, y + 5), (x + 5 + hp_bar_width, y + bar_height + 5), (0, 0, 255), -1)
    cv2.rectangle(frame, (x + 5, y + 5), (x + bar_width + 5, y + bar_height + 5), (150, 150, 150), outline_thickness)

    font_scale = max(0.3, 0.4 * scale)
    cv2.putText(frame, f"HP {hp}/{max_hp}", (x + 10, y + 5 + int(bar_height * 0.9)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), max(1, int(scale)))

    # Mana
    mana_ratio = max(0.0, min(float(mana) / float(max_mana), 1.0))
    mana_bar_width = int(bar_width * mana_ratio)
    cv2.rectangle(frame, (x + 5, y + bar_height + spacing + 5), (x + bar_width + 5, y + bar_height * 2 + spacing + 5), (100, 50, 50), -1)
    cv2.rectangle(frame, (x + 5, y + bar_height + spacing + 5), (x + 5 + mana_bar_width, y + bar_height * 2 + spacing + 5), (255, 0, 0), -1)
    cv2.rectangle(frame, (x + 5, y + bar_height + spacing + 5), (x + bar_width + 5, y + bar_height * 2 + spacing + 5), (150, 150, 150), outline_thickness)
    cv2.putText(frame, f"Mana {mana}/{max_mana}", (x + 10, y + bar_height + spacing + int(bar_height * 0.95)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), max(1, int(scale)))


def main():
    # === QUEST LOG SYSTEM ===
    import random
    QUEST_POOL = [
        "Eat a snack",
        "Ask a mentor for help",
        "Take a group photo",
        "Drink something"
        ]
    q, MAIN_QUEST = generate_mainquest()
    prev_main_quest = MAIN_QUEST
    main_quest_cooldown = False
    main_quest_cooldown_start = 0
    sidequest = random.choice(QUEST_POOL)

    # Initialize quest_log_regions before main loop to avoid empty region on first click
    global quest_log_regions
    quest_box_w = 200
    quest_box_h = 80
    # Assume 1280x720 default if frame not yet available
    default_w, default_h = 1280, 720
    quest_box_x = default_w - quest_box_w - 20
    quest_box_y = 20
    quest_log_regions = []
    main_quest_y1 = quest_box_y + 18
    main_quest_y2 = quest_box_y + 48
    quest_log_regions.append((quest_box_x, main_quest_y1, quest_box_x + quest_box_w, main_quest_y2, 'main', MAIN_QUEST))
    side_quest_y1 = quest_box_y + 52
    side_quest_y2 = quest_box_y + quest_box_h
    quest_log_regions.append((quest_box_x, side_quest_y1, quest_box_x + quest_box_w, side_quest_y2, 'side', sidequest))

    """Initialize models, camera and run the detection loop."""
    parser = argparse.ArgumentParser(description='YOLOv8 RPG View')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML')
    parser.add_argument('--alpha', type=float, default=None, help='Smoothing alpha (overrides config)')
    args = parser.parse_args()

    # Load config if available
    cfg = {}
    if args.config and os.path.exists(args.config):
        if yaml is None:
            print('Warning: PyYAML not installed; skipping config load')
        else:
            try:
                with open(args.config, 'r') as f:
                    cfg = yaml.safe_load(f) or {}
            except Exception as e:
                print(f'Warning: failed to read config {args.config}: {e}')

    # Determine smoothing alpha (CLI overrides config)
    smoothing_alpha = float(cfg.get('smoothing_alpha', SMOOTHING_ALPHA))
    if args.alpha is not None:
        smoothing_alpha = float(args.alpha)

    # Reference distance and HP scale for calibration (can be configured)
    ref_distance_cfg = float(cfg.get('ref_distance', 2.0))
    hp_scale = float(cfg.get('hp_scale', 1.0))
    pending_timeout = float(cfg.get('pending_timeout', 2.5))
    person_timeout = float(cfg.get('person_timeout', 30.0))
    # Minimum bounding-box height (pixels) to consider for assignment/detection
    min_bbox_height = int(cfg.get('min_bbox_height', 40))

    det_model = YOLO("models/yolov8n.pt")

    # Load dragon image once
    dragon_img = None
    dragon_path = os.path.join(os.path.dirname(__file__), '..', 'dragon.jpeg')
    if os.path.exists(dragon_path):
        dragon_img = cv2.imread(dragon_path, cv2.IMREAD_UNCHANGED)
        if dragon_img is not None and dragon_img.shape[2] == 4:
            # Convert to BGR if alpha channel present
            dragon_img = cv2.cvtColor(dragon_img, cv2.COLOR_BGRA2BGR)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Fullscreen window for display (restore correct setup)
    cv2.namedWindow("YOLOv8 RPG View", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("YOLOv8 RPG View", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("YOLOv8 RPG View", mouse_callback)

    import time

    # pending_seen stores detections that haven't been assigned attributes yet:
    # matched_id -> {'first_seen': timestamp, 'last_seen': timestamp, 'bbox': (x1,y1,x2,y2)}
    pending_seen: dict = {}

    # Access globals
    global latest_frame, click_regions

    try:
        boss_id = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # === NEW: UPDATE GLOBAL FRAME ===
            latest_frame = frame.copy() # Save a clean copy for the screenshot
            
            # === NEW: RESET CLICK REGIONS FOR THIS FRAME ===
            current_frame_regions = []

            results = det_model.track(frame, persist=True, classes=[0], conf=0.4, tracker="bytetrack.yaml",verbose=False)

            # Collect UI info for this frame so we can determine the largest hitbox (Boss)
            current_frame_ui: dict = {}
            for r in results:
                if r.boxes.id is None:
                    continue

                for box, track_id in zip(r.boxes, r.boxes.id):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Use upper portion for color/distance/HP
                    full_height = max(1, y2 - y1)
                    # Skip very small detections (likely noise or far away)
                    if full_height < min_bbox_height:
                        continue
                    upper_y2 = y1 + int(full_height * 0.6)
                    upper_height = max(1, upper_y2 - y1)

                    distance = estimate_distance(upper_height)
                    if distance is None:
                        continue

                    attr, matched_id = get_persistent_attributes(track_id, (x1, y1, x2, y2), person_attributes)

                    now = time.time()

                    # initialize hp/mana to safe defaults; they will be set when assigned
                    hp = None
                    mana = None

                    # If we have previous attributes, reuse them and update bbox
                    if attr:
                        hp, mana = attr
                        # update bbox, upper_bbox and last_seen timestamp
                        person_attributes[matched_id].update({
                            'bbox': (x1, y1, x2, y2),
                            'upper_bbox': (x1, y1, x2, upper_y2),
                            'last_seen': now,
                        })
                    else:
                        # Try to match this detection to an existing pending entry (handles tracker id changes)
                        key_id = matched_id
                        # compute IoU helper
                        def _iou(a, b):
                            ax1, ay1, ax2, ay2 = a
                            bx1, by1, bx2, by2 = b
                            ix1 = max(ax1, bx1)
                            iy1 = max(ay1, by1)
                            ix2 = min(ax2, bx2)
                            iy2 = min(ay2, by2)
                            iw = max(0, ix2 - ix1)
                            ih = max(0, iy2 - iy1)
                            inter = iw * ih
                            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
                            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
                            union = area_a + area_b - inter
                            return (inter / union) if union > 0 else 0.0

                        best_iou = 0.0
                        best_pending = None
                        for pid, pinfo in pending_seen.items():
                            pbbox = pinfo.get('bbox')
                            if not pbbox:
                                continue
                            iou = _iou(pbbox, (x1, y1, x2, y2))
                            if iou > best_iou:
                                best_iou = iou
                                best_pending = pid

                        if best_iou > 0.3 and best_pending is not None:
                            key_id = best_pending
                        else:
                            # fallback: centroid proximity to any pending bbox
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            for pid, pinfo in pending_seen.items():
                                ox1, oy1, ox2, oy2 = pinfo.get('bbox', (0, 0, 0, 0))
                                ox = (ox1 + ox2) // 2
                                oy = (oy1 + oy2) // 2
                                dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
                                if dist < PROXIMITY_THRESHOLD:
                                    key_id = pid
                                    break

                        # Track pending visibility duration using the matched key
                        info = pending_seen.get(key_id)
                        if info is None:
                            # start pending timer
                            pending_seen[key_id] = {'first_seen': now, 'last_seen': now, 'bbox': (x1, y1, x2, y2)}
                        else:
                            info['last_seen'] = now
                            info['bbox'] = (x1, y1, x2, y2)

                        # Only assign attributes after the person has been visible for >= pending_timeout
                        visible_time = now - pending_seen[key_id]['first_seen']
                        if visible_time >= pending_timeout:
                            person_upper_crop = frame[y1:upper_y2, x1:x2]
                            if person_upper_crop is None or person_upper_crop.size == 0:
                                # can't measure now; continue and wait for next frame
                                continue
                            # Compute HP from size and Mana from brightness
                            hp = size_to_hp(upper_height, distance, ref_distance=ref_distance_cfg, scale=hp_scale)
                            dominant_bgr = dominant_color(person_upper_crop)
                            mana = color_to_mana(dominant_bgr)
                            # Assign job based on HP/Mana requirements
                            job = assign_job(hp, mana)

                            person_attributes[key_id] = {
                                'hp': hp,
                                'mana': mana,
                                'job': job,
                                'bbox': (x1, y1, x2, y2),
                                'upper_bbox': (x1, y1, x2, upper_y2),
                                'first_seen': pending_seen[key_id]['first_seen'],
                                'assigned_at': now,
                                'last_seen': now,
                            }
                            # no longer pending
                            pending_seen.pop(key_id, None)

                    # === NEW: ADD BOX TO CLICKABLE REGIONS ===
                    # We store the coordinates and ID so the mouse can find them
                    current_frame_regions.append((x1, y1, x2, y2, matched_id))

                    # Draw bounding box (use different color if still pending)
                    if matched_id in person_attributes:
                        stored = person_attributes.get(matched_id)
                        # Determine box color based on job
                        job = stored.get('job', 'Unknown')
                        job_color = JOB_COLORS.get(job, (200, 200, 200))

                        cv2.rectangle(frame, (x1, y1), (x2, y2), job_color, 2)
                        
                        if stored is not None:
                            disp_hp = int(stored.get('hp', 0))
                            disp_mana = int(stored.get('mana', 0))
                            # Compute scaling based on bbox height vs expected pixel height at ref distance
                            expected_px_ref = expected_pixel_height(ref_distance_cfg)
                            raw_scale = 1.0
                            if expected_px_ref > 0:
                                raw_scale = float(full_height) / float(expected_px_ref)
                            # effective scale clamps to [0.5, 1.0]
                            eff_scale = max(0.5, min(1.0, raw_scale))

                            # Center the bars relative to bbox and scale positions
                            bar_x = (x1 + x2) // 2 - int(75 * eff_scale)
                            bar_y = max(10, y1 - int(70 * eff_scale))
                            draw_health_bars(frame, bar_x, bar_y, disp_hp, disp_mana, scale=eff_scale)

                            # Save UI positions and measurements for post-pass boss detection
                            # Calculate area for boss selection
                            area = max(1, (x2 - x1)) * max(1, (y2 - y1))
                            current_frame_ui[matched_id] = {
                                'bar_x': bar_x,
                                'bar_y': bar_y,
                                'job_color': job_color,
                                'eff_scale': eff_scale,
                                'raw_scale': raw_scale,
                                'bbox': (x1, y1, x2, y2),
                                'full_height': full_height,
                                'area': area,
                                'job': job,
                            }

                            # Draw job name above HP/Mana bars only if raw_scale >= 0.5 and not Boss
                            if raw_scale >= 0.5 and matched_id != boss_id:
                                job_text = f"{job}"
                                font_scale = max(0.3, 0.6 * eff_scale)
                                thickness = max(1, int(2 * eff_scale))
                                cv2.putText(frame, job_text, (x1 + 5, max(20, y1 - int(90 * eff_scale))),
                                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, job_color, thickness)
                    else:
                        # pending detection (not yet assigned) - draw gray box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)

            # Determine Boss (largest hitbox) for this frame and draw Boss overlay
            boss_id = None
            if current_frame_ui:
                # Pick the matched_id with largest area
                boss_id = max(current_frame_ui.items(), key=lambda kv: kv[1].get('area', 0))[0]
                boss_info = current_frame_ui.get(boss_id)
                if boss_info:
                    bx1, by1, bx2, by2 = boss_info.get('bbox', (0, 0, 0, 0))
                    eff_scale = boss_info.get('eff_scale', 1.0)
                    raw_scale = boss_info.get('raw_scale', 1.0)
                    bar_x = boss_info.get('bar_x', 0)
                    bar_y = boss_info.get('bar_y', 0)
                    job_color = boss_info.get('job_color', (0, 0, 255))

                    # Draw a thicker bounding box for the Boss
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), max(3, int(3 * eff_scale)))

                    # Draw dragon image and Boss title if raw_scale >= 0.5
                    if raw_scale >= 0.5:
                        panel_bar_w = max(40, int(150 * eff_scale))
                        panel_w = panel_bar_w + 30
                        panel_bar_h = max(8, int(20 * eff_scale))
                        panel_h = panel_bar_h * 2 + 5 * 3

                        # Dragon image to the right of panel
                        if dragon_img is not None:
                            # Resize dragon to fit panel height
                            d_h = panel_h
                            d_w = int(dragon_img.shape[1] * (d_h / dragon_img.shape[0]))
                            dragon_resized = cv2.resize(dragon_img, (d_w, d_h), interpolation=cv2.INTER_AREA)
                            # Compute position
                            dragon_x = bar_x + panel_w + 8
                            dragon_y = bar_y
                            # Overlay dragon image (handle boundaries)
                            y1 = max(0, dragon_y)
                            y2 = min(frame.shape[0], y1 + d_h)
                            x1 = max(0, dragon_x)
                            x2 = min(frame.shape[1], x1 + d_w)
                            # Only overlay if region fits
                            if (y2 > y1) and (x2 > x1):
                                roi = frame[y1:y2, x1:x2]
                                # If dragon_resized has alpha, blend; else direct paste
                                if dragon_resized.shape[2] == 4:
                                    # Blend alpha
                                    alpha = dragon_resized[:, :, 3] / 255.0
                                    for c in range(3):
                                        roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * dragon_resized[:, :, c]
                                else:
                                    roi[:] = dragon_resized[:roi.shape[0], :roi.shape[1], :3]
                        else:
                            # Fallback: draw dragon text
                            dragon_x = bar_x + panel_w + 8
                            dragon_y = bar_y + panel_h // 2 + int(6 * eff_scale)
                            cv2.putText(frame, "DRAGON", (dragon_x, dragon_y), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, 0.6 * eff_scale), (0, 140, 255), max(1, int(2 * eff_scale)))

                        # Display Boss title with person's job (e.g., 'Boss - Mage') above the bars
                        boss_job = boss_info.get('job', 'Unknown')
                        boss_title = f"Boss - {boss_job}"
                        boss_text_x = bx1 + 5
                        boss_text_y = max(20, by1 - int(90 * eff_scale))
                        # Use a bold font for Boss label (thicker outline)
                        cv2.putText(frame, boss_title, (boss_text_x, boss_text_y), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, 0.6 * eff_scale), (0, 215, 255), max(2, int(2 * eff_scale)))

            # === NEW: PUBLISH REGIONS TO GLOBAL ===
            # Update the global list so the mouse callback sees the latest positions
            click_regions = current_frame_regions

            # Cleanup pending_seen entries that haven't been updated recently
            stale_pending = [pid for pid, info in pending_seen.items() if (time.time() - info.get('last_seen', info.get('first_seen', 0))) > pending_timeout]
            for pid in stale_pending:
                pending_seen.pop(pid, None)

            # Cleanup person_attributes entries that haven't been seen recently
            stale_persons = []
            now_cleanup = time.time()
            for pid, data in list(person_attributes.items()):
                last = data.get('last_seen', data.get('assigned_at', now_cleanup))
                if (now_cleanup - last) > person_timeout:
                    stale_persons.append(pid)
            for pid in stale_persons:
                person_attributes.pop(pid, None)


            # Draw Quest Log in top right (with text wrapping)
            import textwrap
            quest_box_w = 220
            quest_box_h = 80
            quest_box_x = frame.shape[1] - quest_box_w - 10
            quest_box_y = 10
            cv2.rectangle(frame, (quest_box_x, quest_box_y), (quest_box_x + quest_box_w, quest_box_y + quest_box_h), (30, 30, 60), -1)
            cv2.rectangle(frame, (quest_box_x, quest_box_y), (quest_box_x + quest_box_w, quest_box_y + quest_box_h), (200, 200, 255), 2)
            # Main Quest
            cv2.putText(frame, "Main Quest:", (quest_box_x + 10, quest_box_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 215, 0), 2)
            if main_quest_cooldown:
                main_lines = textwrap.wrap("No quest...", width=40)
            else:
                main_lines = textwrap.wrap(MAIN_QUEST, width=40)
            for i, line in enumerate(main_lines):
                y = quest_box_y + 32 + i*18
                if y > quest_box_y + 40: break
                cv2.putText(frame, line, (quest_box_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            # Side Quest
            cv2.putText(frame, "Side Quest:", (quest_box_x + 10, quest_box_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 2)
            side_lines = textwrap.wrap(sidequest, width=40)
            for i, line in enumerate(side_lines):
                y = quest_box_y + 65 + i*18
                if y > quest_box_y + quest_box_h - 10: break
                cv2.putText(frame, line, (quest_box_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            # Regenerate sidequest only after success box has shown for 5 seconds
            import time
            if quest_result_overlay['show'] and quest_result_overlay['success'] and quest_result_overlay['quest'] == sidequest:
                elapsed = time.time() - quest_result_overlay['timestamp']
                if elapsed >= 5.0:
                    sidequest = random.choice([q for q in QUEST_POOL if q != sidequest])
                    quest_result_overlay['show'] = False

            # Main quest completion logic
            # Show a big, fancy overlay for 5 seconds before cooldown
            if 'main_quest_success' not in locals():
                main_quest_success = False
                main_quest_success_start = 0

            if quest_result_overlay['show'] and quest_result_overlay['success'] and quest_result_overlay['quest'] == MAIN_QUEST and not main_quest_cooldown and not main_quest_success:
                main_quest_success = True
                main_quest_success_start = time.time()
                quest_result_overlay['show'] = False

            if main_quest_success:
                elapsed_success = time.time() - main_quest_success_start
                if elapsed_success < 5.0:
                    # Draw a big, fancy overlay with 'COMPLETE!' on a new line
                    overlay_w = 600
                    overlay_h = 270
                    ox = frame.shape[1]//2 - overlay_w//2
                    oy = frame.shape[0]//2 - overlay_h//2
                    # Gold border and background
                    cv2.rectangle(frame, (ox, oy), (ox+overlay_w, oy+overlay_h), (40, 40, 10), -1)
                    cv2.rectangle(frame, (ox, oy), (ox+overlay_w, oy+overlay_h), (0, 215, 255), 8)
                    # Confetti (random colored circles)
                    for _ in range(40):
                        cx = ox + np.random.randint(20, overlay_w-20)
                        cy = oy + np.random.randint(20, overlay_h-20)
                        color = tuple(int(x) for x in np.random.choice(range(100,256), size=3))
                        cv2.circle(frame, (cx, cy), np.random.randint(6, 16), color, -1)
                    # Main text split: 'MAIN QUEST' and 'COMPLETE!' on separate lines
                    # Center the main quest success text and quest name
                    main_quest_text = "MAIN QUEST"
                    complete_text = "COMPLETE!"
                    font = cv2.FONT_HERSHEY_TRIPLEX
                    font_scale = 1.7
                    font_thickness = 4
                    # Get text sizes
                    (mq_w, mq_h), _ = cv2.getTextSize(main_quest_text, font, font_scale, font_thickness)
                    (c_w, c_h), _ = cv2.getTextSize(complete_text, font, font_scale, font_thickness)
                    quest_font = cv2.FONT_HERSHEY_SIMPLEX
                    quest_font_scale = .9
                    quest_font_thickness = 3
                    (q_w, q_h), _ = cv2.getTextSize(MAIN_QUEST, quest_font, quest_font_scale, quest_font_thickness)
                    # Calculate centered positions
                    mq_x = ox + (overlay_w - mq_w) // 2
                    mq_y = oy + 90
                    c_x = ox + (overlay_w - c_w) // 2
                    c_y = oy + 150
                    q_x = ox + (overlay_w - q_w) // 2
                    q_y = oy + 210
                    cv2.putText(frame, main_quest_text, (mq_x, mq_y), font, font_scale, (0, 215, 255), font_thickness)
                    cv2.putText(frame, complete_text, (c_x, c_y), font, font_scale, (0, 215, 255), font_thickness)
                    cv2.putText(frame, MAIN_QUEST, (q_x, q_y), quest_font, quest_font_scale, (255,255,255), quest_font_thickness)
                else:
                    main_quest_success = False
                    main_quest_cooldown = True
                    main_quest_cooldown_start = time.time()
                    prev_main_quest = q
                    MAIN_QUEST = ""

            if main_quest_cooldown:
                elapsed_cooldown = time.time() - main_quest_cooldown_start
                if elapsed_cooldown >= 30.0:
                    q_new, new_main_quest = generate_mainquest()
                    if q_new == q:
                        MAIN_QUEST = "Find the Porcelain Throne"
                    else:
                        MAIN_QUEST = new_main_quest
                    main_quest_cooldown = False

            # Update quest_log_regions for click detection (main and side quest clickable areas)
            quest_log_regions = []
            # Main quest region (top half)
            main_quest_y1 = quest_box_y + 18
            main_quest_y2 = quest_box_y + 48
            quest_log_regions.append((quest_box_x, main_quest_y1, quest_box_x + quest_box_w, main_quest_y2, 'main', MAIN_QUEST if not main_quest_cooldown else ""))
            # Side quest region (bottom half)
            side_quest_y1 = quest_box_y + 52
            side_quest_y2 = quest_box_y + quest_box_h
            quest_log_regions.append((quest_box_x, side_quest_y1, quest_box_x + quest_box_w, side_quest_y2, 'side', sidequest))

            # Show quest result overlay if needed
            if quest_result_overlay['show']:
                elapsed = time.time() - quest_result_overlay['timestamp']
                if elapsed < 5.0:
                    # Only show sidequest or failed overlays here
                    overlay_text = ("Quest Complete!" if quest_result_overlay['success'] else "Quest Failed!")
                    color = (0, 255, 0) if quest_result_overlay['success'] else (0, 0, 255)
                    msg = f"{overlay_text}"
                    quest_name = quest_result_overlay['quest']
                    overlay_w = 520
                    overlay_h = 120
                    ox = frame.shape[1]//2 - overlay_w//2
                    oy = frame.shape[0]//2 - overlay_h//2
                    cv2.rectangle(frame, (ox, oy), (ox+overlay_w, oy+overlay_h), (30, 30, 30), -1)
                    cv2.rectangle(frame, (ox, oy), (ox+overlay_w, oy+overlay_h), color, 3)
                    # Center the message and quest name
                    msg_font = cv2.FONT_HERSHEY_SIMPLEX
                    msg_font_scale = 1.2
                    msg_font_thickness = 3
                    quest_font = cv2.FONT_HERSHEY_SIMPLEX
                    quest_font_scale = 0.7
                    quest_font_thickness = 2
                    (msg_w, msg_h), _ = cv2.getTextSize(msg, msg_font, msg_font_scale, msg_font_thickness)
                    (qn_w, qn_h), _ = cv2.getTextSize(quest_name, quest_font, quest_font_scale, quest_font_thickness)
                    msg_x = ox + (overlay_w - msg_w) // 2
                    msg_y = oy + 55
                    qn_x = ox + (overlay_w - qn_w) // 2
                    qn_y = oy + 100
                    cv2.putText(frame, msg, (msg_x, msg_y), msg_font, msg_font_scale, color, msg_font_thickness)
                    cv2.putText(frame, f"{quest_name}", (qn_x, qn_y), quest_font, quest_font_scale, (255,255,255), quest_font_thickness)
                else:
                    quest_result_overlay['show'] = False

            cv2.imshow("YOLOv8 RPG View", frame)
            # Correct quit logic: waitKey returns int, mask with 0xFF
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Show loading screen for 10 seconds before running main
    import cv2, time, os
    loading_path = os.path.join(os.path.dirname(__file__), '..', 'loading.png')
    if not os.path.exists(loading_path):
        loading_path = os.path.join(os.path.dirname(__file__), 'loading.png')
    img = cv2.imread(loading_path)
    if img is not None:
        # Get screen size
        try:
            import ctypes
            user32 = ctypes.windll.user32
            screen_w, screen_h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        except Exception:
            screen_w, screen_h = 1920, 1080  # fallback
        img_h, img_w = img.shape[:2]
        scale = min(screen_w / img_w, screen_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        # Resize image
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Create black background
        fullscreen = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        # Center the image
        x_offset = (screen_w - new_w) // 2
        y_offset = (screen_h - new_h) // 2
        fullscreen[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
        cv2.namedWindow("Loading...", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Loading...", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Loading...", fullscreen)
        cv2.waitKey(1)
        start = time.time()
        while time.time() - start < 10:
            if cv2.getWindowProperty("Loading...", cv2.WND_PROP_VISIBLE) < 1:
                break
            cv2.imshow("Loading...", fullscreen)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        cv2.destroyWindow("Loading...")
    main()
