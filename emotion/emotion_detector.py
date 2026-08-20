import time
import random
import threading

try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None

class EmotionDetector:
    def __init__(self, config):
        self.config = config
        # Cache to store last detected emotion per tracker ID: tracker_id -> emotions_dict
        self.emotion_cache = {}
        # Timestamps of last analyses: tracker_id -> float timestamp
        self.last_update_times = {}
        # Set of active background threads: tracker_id
        self.running_analyses = set()
        # Throttled update interval: start a background analysis thread at most every 2.5 seconds per person
        self.throttle_interval = 2.5

    def analyze_emotion(self, face_frame, tracker_id):
        if not self.config.is_module_enabled("emotion_detection"):
            return None
            
        current_time = time.time()
        
        # If there is already an active thread running DeepFace for this person, return last cache immediately
        if tracker_id in self.running_analyses:
            return self.emotion_cache.get(tracker_id, {"Neutral": 100.0})
            
        # Check if we need to schedule a new background analysis
        last_time = self.last_update_times.get(tracker_id, 0.0)
        if current_time - last_time >= self.throttle_interval:
            self.running_analyses.add(tracker_id)
            
            # Start background thread to run inference without locking the main thread
            thread = threading.Thread(
                target=self._async_analysis_worker,
                args=(face_frame.copy(), tracker_id, current_time),
                daemon=True
            )
            thread.start()
            
        # Instantly return the cached emotions (or a default neutral layout if it's the first frame)
        return self.emotion_cache.get(tracker_id, {"Neutral": 100.0})

    def _async_analysis_worker(self, face_crop, tracker_id, current_time):
        try:
            if DeepFace is None or face_crop.size == 0:
                emotions = self._generate_mock_emotions(tracker_id)
            else:
                # DeepFace analyze on the crop
                analysis = DeepFace.analyze(
                    img_path=face_crop, 
                    actions=['emotion'], 
                    enforce_detection=False,
                    silent=True
                )
                
                if isinstance(analysis, list):
                    result = analysis[0]
                else:
                    result = analysis
                    
                raw_emotions = result.get("emotion", {})
                emotions = {k.capitalize(): float(v) for k, v in raw_emotions.items()}
                
            # Update cache and timestamp
            self.emotion_cache[tracker_id] = emotions
            self.last_update_times[tracker_id] = current_time
        except Exception as e:
            # Fallback to mock on error to prevent crashes
            emotions = self._generate_mock_emotions(tracker_id)
            self.emotion_cache[tracker_id] = emotions
            self.last_update_times[tracker_id] = current_time
        finally:
            # Always clean active thread tracking flag
            if tracker_id in self.running_analyses:
                self.running_analyses.remove(tracker_id)

    def _generate_mock_emotions(self, tracker_id):
        # Deterministic mock emotions based on tracker ID to make simulation look consistent
        random.seed(tracker_id + 42)
        emotions_list = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Fear", "Disgust"]
        
        # Generate random weights
        weights = [random.random() for _ in emotions_list]
        dominant_idx = random.choice([0, 0, 0, 1, 2]) # 60% neutral, 20% happy, 20% surprise
        weights[dominant_idx] += 2.0
        new_total = sum(weights)
        
        normalized = {}
        for emotion, w in zip(emotions_list, weights):
            normalized[emotion] = round((w / new_total) * 100.0, 1)
            
        return normalized

    def cleanup_tracker(self, tracker_id):
        if tracker_id in self.emotion_cache:
            del self.emotion_cache[tracker_id]
        if tracker_id in self.last_update_times:
            del self.last_update_times[tracker_id]
        if tracker_id in self.running_analyses:
            self.running_analyses.remove(tracker_id)
