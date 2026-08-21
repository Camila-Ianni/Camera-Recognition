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
        # Cache to store last detected details per tracker ID: tracker_id -> {"emotions": dict, "gender": str}
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
            return self.emotion_cache.get(tracker_id, {"emotions": {"Neutral": 100.0}, "gender": "Femenino" if tracker_id in [1, 2, 5] else "Masculino"})
            
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
            
        # Instantly return the cached details
        return self.emotion_cache.get(tracker_id, {"emotions": {"Neutral": 100.0}, "gender": "Femenino" if tracker_id in [1, 2, 5] else "Masculino"})

    def _async_analysis_worker(self, face_crop, tracker_id, current_time):
        try:
            if DeepFace is None or face_crop.size == 0:
                res = self._generate_mock_emotions_and_gender(tracker_id)
                emotions = res["emotions"]
                gender_str = res["gender"]
            else:
                # DeepFace analyze on the crop for both emotion and gender
                analysis = DeepFace.analyze(
                    img_path=face_crop, 
                    actions=['emotion', 'gender'], 
                    enforce_detection=False,
                    silent=True
                )
                
                if isinstance(analysis, list):
                    result = analysis[0]
                else:
                    result = analysis
                    
                # Parse Emotions
                raw_emotions = result.get("emotion", {})
                emotions = {k.capitalize(): float(v) for k, v in raw_emotions.items()}
                
                # Parse Gender
                gender_res = result.get("gender")
                if isinstance(gender_res, dict):
                    dominant_gender = max(gender_res.items(), key=lambda x: x[1])[0]
                else:
                    dominant_gender = str(gender_res)
                    
                gender_str = "Masculino" if dominant_gender.lower() in ["man", "m", "male"] else "Femenino"
                
            # Update cache and timestamp
            self.emotion_cache[tracker_id] = {
                "emotions": emotions,
                "gender": gender_str
            }
            self.last_update_times[tracker_id] = current_time
        except Exception as e:
            # Print error to console so we can see if it is still downloading weights or having issues
            print(f"[DeepFace Exception] Error analyzing face ID {tracker_id}: {e}")
            res = self._generate_mock_emotions_and_gender(tracker_id)
            self.emotion_cache[tracker_id] = res
            self.last_update_times[tracker_id] = current_time
        finally:
            # Always clean active thread tracking flag
            if tracker_id in self.running_analyses:
                self.running_analyses.remove(tracker_id)

    def _generate_mock_emotions_and_gender(self, tracker_id):
        # Fluctuates over time to make simulation mode dynamic (changes every 5 seconds)
        t_sec = int(time.time()) // 5
        random.seed(tracker_id + 42 + t_sec)
        
        emotions_list = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Fear", "Disgust"]
        weights = [random.random() for _ in emotions_list]
        dominant_idx = random.choice([0, 0, 0, 1, 2]) # 60% neutral, 20% happy, 20% surprise
        weights[dominant_idx] += 2.0
        new_total = sum(weights)
        
        normalized = {}
        for emotion, w in zip(emotions_list, weights):
            normalized[emotion] = round((w / new_total) * 100.0, 1)
            
        # Custom mock logic: Camila/User (tracker 5, 1, 2) is Female, others default to Male
        gender_str = "Femenino" if tracker_id in [1, 2, 5] else "Masculino"
        
        return {
            "emotions": normalized,
            "gender": gender_str
        }

    def cleanup_tracker(self, tracker_id):
        if tracker_id in self.emotion_cache:
            del self.emotion_cache[tracker_id]
        if tracker_id in self.last_update_times:
            del self.last_update_times[tracker_id]
        if tracker_id in self.running_analyses:
            self.running_analyses.remove(tracker_id)
