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

#=======the main class and functions for the quest workflow======

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
            return
            
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





#========Implementation into main func ========

#====== after while cap.isOpened() is called ====== 

#======== after for pid in stale persons =======
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