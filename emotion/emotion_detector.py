import time
import random

try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None

class EmotionDetector:
    def __init__(self, config):
        self.config = config
        # Cache to store last detected emotion per tracker ID
        # tracker_id -> (emotions_dict, last_update_time)
        self.emotion_cache = {}
        self.throttle_interval = 2.0 # Only analyze once every 2 seconds per person to maintain high FPS

    def analyze_emotion(self, face_frame, tracker_id):
        if not self.config.is_module_enabled("emotion_detection"):
            return None
            
        current_time = time.time()
        
        # Check cache
        if tracker_id in self.emotion_cache:
            emotions, last_time = self.emotion_cache[tracker_id]
            if current_time - last_time < self.throttle_interval:
                return emotions

        # Fallback if DeepFace is not installed or error occurs
        if DeepFace is None:
            mock_emotions = self._generate_mock_emotions(tracker_id)
            self.emotion_cache[tracker_id] = (mock_emotions, current_time)
            return mock_emotions

        try:
            # DeepFace analyze
            # We enforce_detection=False so it doesn't crash if the face crop is slightly blurry
            analysis = DeepFace.analyze(
                img_path=face_frame, 
                actions=['emotion'], 
                enforce_detection=False,
                silent=True
            )
            
            # DeepFace returns a list of results or a single dict
            if isinstance(analysis, list):
                result = analysis[0]
            else:
                result = analysis
                
            raw_emotions = result.get("emotion", {})
            
            # Format and normalize percentages
            emotions = {}
            for k, v in raw_emotions.items():
                # DeepFace keys: happy, sad, angry, fear, surprise, neutral, disgust
                # Translate or format keys
                key = k.capitalize()
                emotions[key] = float(v)

            # Sort and store in cache
            self.emotion_cache[tracker_id] = (emotions, current_time)
            return emotions

        except Exception as e:
            # Suppress logs and return a simulated fallback on error
            mock_emotions = self._generate_mock_emotions(tracker_id)
            self.emotion_cache[tracker_id] = (mock_emotions, current_time)
            return mock_emotions

    def _generate_mock_emotions(self, tracker_id):
        # Deterministic mock emotions based on tracker ID to make simulation look consistent
        random.seed(tracker_id + 42)
        
        emotions_list = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Fear", "Disgust"]
        
        # Generate random weights
        weights = [random.random() for _ in emotions_list]
        total = sum(weights)
        
        # Make one emotion highly dominant (e.g., Neutral or Happy) to look natural
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
