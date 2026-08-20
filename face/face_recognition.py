import cv2
import numpy as np

try:
    import face_recognition
except ImportError:
    face_recognition = None

class FaceRecognizer:
    def __init__(self, face_db, tolerance=0.5):
        self.face_db = face_db
        self.tolerance = tolerance

    def recognize_faces(self, frame):
        # Returns list of dicts: {"box": (y1, x2, y2, x1), "name": str, "status": str, "distance": float}
        results = []
        
        if face_recognition is None:
            # Fallback mock facial recognition
            # We can return a mock face detection if we want, or do nothing.
            # In live webcam mode, if face_recognition is not compiled, we can mock it
            # based on any detected faces.
            return results

        try:
            # Downscale frame to 1/4 size for 16x speedup
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find all face locations and encodings in the frame
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            if not face_locations:
                return results

            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            known_encodings, known_metadata = self.face_db.get_known_faces_cache()
            
            for face_loc, face_enc in zip(face_locations, face_encodings):
                top, right, bottom, left = face_loc
                # Scale back up by 4x
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                name = "PERSONA DESCONOCIDA"
                status = "unknown"
                min_dist = 1.0
                
                if known_encodings:
                    # Compare face encodings
                    matches = face_recognition.compare_faces(known_encodings, face_enc, tolerance=self.tolerance)
                    distances = face_recognition.face_distance(known_encodings, face_enc)
                    
                    if True in matches:
                        best_match_idx = np.argmin(distances)
                        if matches[best_match_idx]:
                            min_dist = float(distances[best_match_idx])
                            meta = known_metadata[best_match_idx]
                            name = meta["name"]
                            status = meta["status"] # authorized / unauthorized
                
                # face_loc is (top, right, bottom, left)
                results.append({
                    "box": (top, right, bottom, left),
                    "name": name,
                    "status": status,
                    "confidence": 1.0 - min_dist
                })
        except Exception as e:
            print(f"Error during face recognition: {e}")
            
        return results

    def mock_recognize(self, person_tracker_id):
        # Simulated face recognition for testing when no cameras or library is compiled
        # Camila Ianni is authorized, other odd IDs are unauthorized, even IDs are unknown
        if person_tracker_id == 1:
            return "Camila Ianni", "authorized", 0.92
        elif person_tracker_id % 3 == 0:
            return "Persona de Alerta", "unauthorized", 0.88
        else:
            return "PERSONA DESCONOCIDA", "unknown", 0.0
