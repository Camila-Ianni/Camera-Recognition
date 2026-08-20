import cv2
import time
import os

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class ObjectDetector:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.load_model()

    def load_model(self):
        if YOLO is None:
            print("Warning: ultralytics (YOLO) is not available. Object detection is running in simulated mode.")
            return False

        try:
            # Load YOLOv8 Nano model (weights are downloaded automatically if missing)
            # Ensure model directory exists
            os.makedirs("models", exist_ok=True)
            self.model = YOLO("models/yolov8n.pt")
            print("YOLOv8 model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None
            return False

    def detect(self, frame):
        # Returns (people_bboxes, other_objects)
        # people_bboxes: list of (x1, y1, x2, y2)
        # other_objects: list of dicts {"class_name": str, "confidence": float, "box": (x1, y1, x2, y2)}
        people_bboxes = []
        other_objects = []
        
        if not self.config.is_module_enabled("object_detection"):
            return people_bboxes, other_objects

        min_conf = self.config.get_min_confidence()

        if self.model is None:
            # Simulated Mock Detections
            return self._mock_detect(frame)

        try:
            # Run inference
            # verbose=False reduces terminal spam
            results = self.model(frame, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < min_conf:
                        continue
                        
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    
                    # Bounding box coordinates
                    # xyxy is a tensor, we fetch coordinates
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = [int(c) for c in coords]
                    
                    if class_name == "person":
                        people_bboxes.append((x1, y1, x2, y2))
                    else:
                        other_objects.append({
                            "class_name": class_name,
                            "confidence": conf,
                            "box": (x1, y1, x2, y2)
                        })
        except Exception as e:
            print(f"Error during YOLO inference: {e}")
            # Fallback to mock on error
            return self._mock_detect(frame)
            
        return people_bboxes, other_objects

    def _mock_detect(self, frame):
        # Mock detection simulation for demonstration purposes
        # When a camera captures frames, we can simulate detections based on timestamps
        people = []
        objects = []
        
        h, w, _ = frame.shape
        t = time.time()
        
        # Simulate 1 or 2 persons moving slowly in the frame
        # Person 1
        x_center = int(w * (0.3 + 0.15 * (1 + int(t / 2) % 10) / 10.0))
        y_center = int(h * 0.5)
        pw, ph = int(w * 0.18), int(h * 0.6)
        px1 = max(0, x_center - pw // 2)
        py1 = max(0, y_center - ph // 2)
        px2 = min(w, x_center + pw // 2)
        py2 = min(h, y_center + ph // 2)
        
        people.append((px1, py1, px2, py2))
        
        # Simulate a static laptop or cell phone
        objects.append({
            "class_name": "laptop",
            "confidence": 0.88,
            "box": (int(w * 0.45), int(h * 0.65), int(w * 0.58), int(h * 0.82))
        })
        
        # Occasionally simulate a risk object (like a scissors or backpack) to trigger alerts
        # Every 25 seconds, simulate a risk object for 5 seconds
        cycle = int(t) % 25
        if 8 <= cycle <= 13:
            objects.append({
                "class_name": "backpack",
                "confidence": 0.74,
                "box": (int(w * 0.75), int(h * 0.5), int(w * 0.88), int(h * 0.75))
            })
        elif 18 <= cycle <= 23:
            objects.append({
                "class_name": "scissors",
                "confidence": 0.81,
                "box": (int(w * 0.22), int(h * 0.4), int(w * 0.28), int(h * 0.48))
            })
            
        return people, objects
