import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

class PersonLabelAssigner:
    """Assigns labels to detected people based on clothing color."""
    
    def __init__(self):
        # Color ranges in HSV (Hue, Saturation, Value)
        # Hue: 0-180 (in OpenCV), Saturation: 0-255, Value: 0-255
        self.color_ranges = {
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
        
        # Color to label mapping
        self.label_mapping = {
            'red': 'enemy',
            'blue': 'friend',
            'green': 'friend',
            'yellow': 'romance target',
            'purple': 'enemy',
            'black': 'friend'
        }
    
    def detect_dominant_color(self, frame, x1, y1, x2, y2):
        """Detect the dominant color in a bounding box region."""
        # Extract the region of interest (clothing area, focusing on upper body)
        roi = frame[y1:y2, x1:x2]
        
        # Skip if ROI is too small
        if roi.shape[0] < 10 or roi.shape[1] < 10:
            return None
        
        # Focus on upper portion of bounding box (shoulders/chest)
        height = roi.shape[0]
        roi = roi[:int(height * 0.6), :]
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        color_counts = defaultdict(int)
        
        # Check each color range
        for color, ranges in self.color_ranges.items():
            if color == 'red':
                # Red wraps around in HSV
                mask1 = cv2.inRange(hsv, ranges['lower1'], ranges['upper1'])
                mask2 = cv2.inRange(hsv, ranges['lower2'], ranges['upper2'])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, ranges['lower'], ranges['upper'])
            
            # Count pixels of this color
            color_counts[color] = cv2.countNonZero(mask)
        
        # Return the color with the most pixels
        if max(color_counts.values()) > 0:
            return max(color_counts, key=color_counts.get)
        return None
    
    def get_label(self, detected_color):
        """Get the label for a detected color."""
        if detected_color and detected_color in self.label_mapping:
            return self.label_mapping[detected_color]
        return 'unknown'


def main():
    """Main function to run real-time person detection and labeling."""
    
    # Load YOLOv8 model
    model = YOLO('models/yolov8n.pt')  # nano model for faster inference
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # Initialize label assigner
    assigner = PersonLabelAssigner()
    
    print("Starting person detection and labeling...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to read frame")
            break
        
        # Run YOLOv8 inference
        results = model(frame, conf=0.5, classes=0)  # class 0 is 'person' in COCO dataset
        
        # Process detections
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Detect clothing color
                detected_color = assigner.detect_dominant_color(frame, x1, y1, x2, y2)
                
                # Get label based on color
                label = assigner.get_label(detected_color)
                
                # Determine box color based on label
                if label == 'enemy':
                    box_color = (0, 0, 255)  # Red
                elif label == 'friend':
                    box_color = (0, 255, 0)  # Green
                elif label == 'romance target':
                    box_color = (0, 255, 255)  # Yellow
                else:
                    box_color = (128, 128, 128)  # Gray
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Prepare label text
                label_text = f"{label} ({detected_color})"
                confidence_text = f"Conf: {conf:.2f}"
                
                # Get text size for background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                
                (text_width, text_height) = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
                
                # Draw label background
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_height - 20),
                    (x1 + text_width + 10, y1),
                    box_color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    frame,
                    label_text,
                    (x1 + 5, y1 - 10),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness
                )
                
                # Draw confidence below
                cv2.putText(
                    frame,
                    confidence_text,
                    (x1 + 5, y2 + 20),
                    font,
                    0.5,
                    box_color,
                    1
                )
        
        # Display frame
        cv2.imshow('YOLOv8 Person Detection and Labeling', frame)
        
        # Check for 'q' key press to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped")


if __name__ == '__main__':
    main()
