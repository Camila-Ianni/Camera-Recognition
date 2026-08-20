import os
import cv2
import time
import threading
from collections import deque
from datetime import datetime

class EvidenceManager:
    def __init__(self, buffer_duration=2, fps=15):
        # buffer_duration: seconds of video before the trigger
        self.fps = fps
        self.buffer_size = int(buffer_duration * fps)
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.lock = threading.Lock()

    def add_frame(self, frame):
        if frame is None:
            return
        with self.lock:
            # Save a scaled down copy or full size frame
            self.frame_buffer.append(frame.copy())

    def trigger_evidence_capture(self, camera_manager, event_id, event_name, completion_callback=None):
        # event_name: e.g. "objeto_de_riesgo" or "persona_desconocida"
        # Gather the "before" frames
        with self.lock:
            pre_event_frames = list(self.frame_buffer)
            
        # Spawn a thread to collect "after" frames and write the video
        thread = threading.Thread(
            target=self._capture_evidence_worker,
            args=(camera_manager, event_id, event_name, pre_event_frames, completion_callback),
            daemon=True
        )
        thread.start()

    def _capture_evidence_worker(self, camera_manager, event_id, event_name, pre_event_frames, completion_callback):
        # Collect post-event frames for the next 2 seconds
        post_event_frames = []
        post_frame_count = self.buffer_size
        
        # Read from camera manager at the specified rate
        delay = 1.0 / self.fps
        for _ in range(post_frame_count):
            frame = camera_manager.get_frame()
            if frame is not None:
                post_event_frames.append(frame)
            time.sleep(delay)

        # Merge sequences
        all_frames = pre_event_frames + post_event_frames
        if not all_frames:
            print("Evidence capture failed: No frames gathered.")
            return

        # Prepare folder and names
        os.makedirs("data/alerts", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_name = event_name.lower().replace(" ", "_")
        
        video_filename = f"data/alerts/{timestamp}_{event_id}_{safe_name}.mp4"
        image_filename = f"data/alerts/{timestamp}_{event_id}_{safe_name}.jpg"

        # 1. Save trigger thumbnail (usually the middle frame, or trigger point)
        trigger_index = len(pre_event_frames) - 1 if pre_event_frames else 0
        trigger_frame = all_frames[min(trigger_index, len(all_frames)-1)]
        cv2.imwrite(image_filename, trigger_frame)

        # 2. Write MP4 video file
        height, width, _ = all_frames[0].shape
        
        # OpenCV VideoWriter setup
        # mp4v is standard and works well on Mac / Windows
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_filename, fourcc, self.fps, (width, height))
        
        for frame in all_frames:
            out.write(frame)
        out.release()
        
        print(f"Evidence saved: {video_filename} and thumbnail: {image_filename}")
        
        # Trigger database update or other callback
        if completion_callback:
            completion_callback(event_id, video_filename)
