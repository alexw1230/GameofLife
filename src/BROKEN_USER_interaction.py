import argparse
import os
import random
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from pathlib import Path
from openai import OpenAI
import time

try:
    import yaml
except Exception:
    yaml = None

client = OpenAI(
    api_key="APIKEY",
    base_url="https://api.featherless.ai/v1"
)
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

def mouse_callback(event, x, y, flags, param):
    """Handles mouse clicks on the video window."""
    global latest_frame, click_regions


    #======NEW for handling checking for quest completion=====
    global  workflow
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if we are currently trying to complete a quest
        if workflow.waiting_for_click:
            if latest_frame is not None:
                workflow.handle_capture(latest_frame)
            return
    #====end of new check =====
    # Trigger only on Left Mouse Button Click (LBUTTONDOWN)
    if event == cv2.EVENT_LBUTTONDOWN:
        
        # Check if the click (x, y) is inside any of the detected boxes
        for (x1, y1, x2, y2, track_id) in click_regions:
            if x1 < x < x2 and y1 < y < y2:
                print(f"✅ Clicked on Person ID: {track_id}")
                
                if latest_frame is not None:
                    # --- TAKE THE SCREENSHOT (CROP) ---
                    # Ensure we don't crop outside the image
                    h, w, _ = latest_frame.shape
                    crop_x1, crop_y1 = max(0, x1), max(0, y1)
                    crop_x2, crop_y2 = min(w, x2), min(h, y2)
                    
                    person_crop = latest_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    # --- OPEN THE NEW WINDOW ---
                    if person_crop.size > 0:
                        window_name = f"Profile: Person {track_id}"
                        cv2.imshow(window_name, person_crop)
                        print(f"   -> Opened window: {window_name}")
                return # Stop checking after finding the first match





# =================== Persistent attributes ===================
# track_id -> {'hp': HP, 'mana': Mana, 'job': job, 'bbox': (x1,y1,x2,y2), 'upper_bbox': (...), ...}
person_attributes = {}


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





#======quest code=======

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



#===== end of quest code



#=====quest workflow code, NEW =======
#first simple ui design, design click here box
# should occur at first entry but 
#def draw_complete_quest_button(frame, quest_manager):
    #cv2.putText(frame, "[Press M for Menu]", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
class QuestWorkflow:
    def __init__(self, check_func):
        self.active = False  # Is the quest menu open?
        self.status_message = ""
        self.message_expiry = 0
        self.last_result = None # True for success, False for fail
#=====added bulll========
        self.menu_active = False # maybe redundant
        self.selected_idx = 0
        self.check_func = check_func  # This links to your check_quest_complete
        self.feedback_msg = None
        self.feedback_expiry = 0
        #even more 
        self.waiting_for_click = False  # New state for mouse capture
        
        # Hidden identifiers for the AI
        self.available_quests = [
            {"ui_name": "Gather Herbs", "id": "a green plant or leaf"},
            {"ui_name": "Identify a Mage", "id": "a person wearing blue or a wizard hat"},
            {"ui_name": "Find a Health Potion", "id": "a red liquid or red bottle"},
            {"ui_name": "Locate a Beast", "id": "a dog, cat, or animal"}
        ]
        self.selected_index = 0

    def draw_hud(self, frame):
        h, w, _ = frame.shape
        
        # 1. The "Complete Task" Trigger Box (Top Left)
        cv2.rectangle(frame, (20, 20), (220, 70), (0, 0, 0), -1)
        cv2.rectangle(frame, (20, 20), (220, 70), (0, 255, 0), 2)
        cv2.putText(frame, " [TAB] QUESTS", (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 2. The Interactive Selection Window
        if self.active:
            overlay = frame.copy()
            mw, mh = 400, 300
            cx, cy = w // 2 - mw // 2, h // 2 - mh // 2
            cv2.rectangle(overlay, (cx, cy), (cx + mw, cy + mh), (30, 30, 30), -1)
            cv2.rectangle(overlay, (cx, cy), (cx + mw, cy + mh), (255, 255, 255), 2)
            
            cv2.putText(overlay, "SELECT YOUR QUEST", (cx + 80, cy + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 2)
            
            for i, q in enumerate(self.available_quests):
                color = (0, 255, 0) if i == self.selected_index else (150, 150, 150)
                prefix = "> " if i == self.selected_index else "  "
                cv2.putText(overlay, f"{prefix}{q['ui_name']}", (cx + 30, cy + 100 + (i * 40)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.putText(overlay, "[SPACE] COMPLETE   [W/S] NAVIGATE", (cx + 50, cy + 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
        # 3. Success/Failure Feedback (Thumbs Up/Down)
        if time.time < self.message_expiry:
            icon = "SUCCESS! (Y)" if self.last_result else "FAILED (N)"
            icon_color = (0, 255, 0) if self.last_result else (0, 0, 255)
            cv2.putText(frame, icon, (w//2 - 100, h//2), cv2.FONT_HERSHEY_TRIPLEX, 2, icon_color, 4)

#====CHANGEs TO class=====
    def proccess_input(self, menu_key, current_frame):
        if menu_key == ord('m') or menu_key == ord('M'):
            self.menu_active = not self.menu_active
            self.waiting_for_click = False # Reset if toggled
            
            
            # --- Handle Workflow Inputs ---
        #if menu_key == 9: # TAB key
            # workflow.active = not workflow.active
            
        if workflow.active:
                if menu_key == ord('w'):
                    workflow.selected_index = (workflow.selected_index - 1) % len(workflow.available_quests)
                elif menu_key == ord('s'):
                 workflow.selected_index = (workflow.selected_index + 1) % len(workflow.available_quests)
                elif menu_key == ord(' '): # SPACE to take picture and verify
                # Save current frame temporarily
                    self.menu_active = False
                    self.waiting_for_click = True # New state for mouse capture

                    #cv2.imwrite("quest_proof.jpg", current_frame)
                
                # Use your AI function

                
    def draw(self, frame):
        h, w, _ = frame.shape
        # Draw Feedback (Thumbs Up/Down logic)
        if self.feedback_msg and time.time() < self.feedback_expiry:
            color = (0, 255, 0) if self.feedback_msg == "SUCCESS" else (0, 0, 255)
            # Simple text representation of Thumbs Up/Down
            label = "(Y) SUCCESS" if self.feedback_msg == "SUCCESS" else "(N) FAILED"
            cv2.putText(frame, label, (w//2-100, h//2), cv2.FONT_HERSHEY_TRIPLEX, 2, color, 4)

        # Draw Selection Menu
        if self.menu_active:
            overlay = frame.copy()
            cv2.rectangle(overlay, (w//2-200, h//2-150), (w//2+200, h//2+100), (40, 40, 40), -1)
            cv2.putText(overlay, "SELECT QUEST", (w//2-100, h//2-110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            for i, q in enumerate(self.quest_list):
                color = (0, 255, 255) if i == self.selected_idx else (150, 150, 150)
                prefix = "> " if i == self.selected_idx else "  "
                cv2.putText(overlay, f"{prefix}{q['name']}", (w//2-170, h//2-50 + (i*40)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.putText(overlay, "[SPACE] Fulfill Quest", (w//2-100, h//2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

            '''
                # Set feedback state
            workflow.last_result = success
            workflow.message_expiry = time.time() + 3 # Show for 3 seconds
            workflow.active = False # Close menu
                '''
               
def handle_capture(self, image_to_save):
        """This is called when the mouse clicks during 'waiting_for_click' state."""
        self.waiting_for_click = False
        
        # Save screenshot
        filename = "quest_proof.jpg"
        cv2.imwrite(filename, image_to_save)
        print(f"Captured {filename}. Sending to AI...")


        target_id = self.quest_list[self.selected_idx]['hidden_id'] # of debatable value 
        success = self.check_func("quest_proof.jpg", self.quest_list[self.selected_idx]['hidden_id'])
                
        self.feedback_msg = "SUCCESS" if success else "FAILED"
        self.feedback_expiry = time.time() + 3.0 # Show for 3 seconds


#workflow = QuestWorkflow()

workflow = QuestWorkflow(check_quest_complete)
#============END OF NEW=======




##========MAIN==========
def main():
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
        while cap.isOpened(): #======ACTUAL MAIN========
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
                    current_frame_regions.append((x1, y1, x2, y2, int(track_id)))

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
                            current_frame_ui[matched_id] = {
                                'bar_x': bar_x,
                                'bar_y': bar_y,
                                'job_color': job_color,
                                'eff_scale': eff_scale,
                                'raw_scale': raw_scale,
                                'bbox': (x1, y1, x2, y2),
                                'full_height': full_height,
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
                # Pick the matched_id with largest full_height
                boss_id = max(current_frame_ui.items(), key=lambda kv: kv[1].get('full_height', 0))[0]
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
                        cv2.putText(frame, boss_title, (boss_text_x, boss_text_y), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, 0.6 * eff_scale), (0, 215, 255), max(1, int(2 * eff_scale)))

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
            '''
            #======= NEW =======
            workflow.draw_hud(frame)
            cv2.imshow("YOLOv8 RPG View", frame)
            menu_key = cv2.waitKey(1) & 0xFF
            # --- Handle Workflow Inputs ---
            if menu_key == 9: # TAB key
             workflow.active = not workflow.active
            
            if workflow.active:
                if menu_key == ord('w'):
                    workflow.selected_index = (workflow.selected_index - 1) % len(workflow.available_quests)
                elif menu_key == ord('s'):
                 workflow.selected_index = (workflow.selected_index + 1) % len(workflow.available_quests)
                elif menu_key == ord(' '): # SPACE to take picture and verify
                # Save current frame temporarily
                    temp_path = "quest_capture.jpg"
                    cv2.imwrite(temp_path, frame)
                
                # Run the AI check against the hidden ID
                current_quest_id = workflow.available_quests[workflow.selected_index]['id']
                print(f"Submitting proof for: {current_quest_id}...")
                
                success = check_quest_complete(temp_path, current_quest_id)
                
                # Set feedback state
                workflow.last_result = success
                workflow.message_expiry = time.time() + 3 # Show for 3 seconds
                workflow.active = False # Close menu

                if menu_key == 27:
                    break
        '''
        





            #====== END OF NEW =====




            #===== new again ====

            if not workflow.menu_active:
                cv2.putText(frame, "[M] Quest Menu", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
            workflow.draw(frame)

            #cv2.imshow("YOLOv8 RPG View", frame)
    
            key = cv2.waitKey(1) & 0xFF
             # 3. Pass keys to the workflow
            workflow.proccess_input(key, frame)
    
            if key == 27: # ESC
                break
            #draw_quest_ui(frame, quest_sys)
            

            cv2.imshow("YOLOv8 RPG View", frame)
            # Correct quit logic: waitKey returns int, mask with 0xFF
            #key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'): # Press 'R' to shuffle quest
                quest_sys.refresh_side_quest()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
