import cv2
import time
import threading
import numpy as np

class CameraManager:
    def __init__(self, source=0):
        self.source = source
        self.cap = None
        self.frame = None
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # FPS Calculation
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Flag to indicate if we have a new frame
        self.has_new_frame = False

    def start(self):
        if self.is_running:
            return True
            
        # Try opening the video capture
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera source {self.source}")
            return False
            
        # Set hardware webcam dimensions explicitly to prevent buffer stride glitches
        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.frame = None

    def change_source(self, new_source):
        self.stop()
        self.source = new_source
        return self.start()

    def _capture_loop(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                # If we are reading a video file (simulation), loop it
                if isinstance(self.source, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.03) # avoid infinite tight loop on EOF error
                    continue
                else:
                    print("Camera read failed, retrying...")
                    time.sleep(1.0)
                    continue

            with self.lock:
                self.frame = frame
                self.has_new_frame = True

            # FPS calculation
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = time.time()
                
            # Sleep briefly to avoid maxing out CPU
            time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                self.has_new_frame = False
                return self.frame.copy()
            return None

    def get_fps(self):
        return round(self.fps, 1)

    def is_new_frame_available(self):
        with self.lock:
            return self.has_new_frame
