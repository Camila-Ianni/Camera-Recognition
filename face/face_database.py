import os
import cv2
import pickle
import numpy as np

# We import face_recognition inside functions or with a try-except block 
# to ensure the system doesn't crash on import if compilation is still running.
try:
    import face_recognition
except ImportError:
    face_recognition = None

class FaceDatabase:
    def __init__(self, db_manager):
        self.db = db_manager
        self.known_encodings = []  # list of numpy arrays
        self.known_metadata = []   # list of dicts with name, status, identifier
        
        self.load_known_faces()

    def load_known_faces(self):
        self.known_encodings = []
        self.known_metadata = []
        
        if face_recognition is None:
            print("Warning: face_recognition is not available. Facial recognition is disabled.")
            return

        people = self.db.get_registered_people()
        for person in people:
            img_path = person["reference_image_path"]
            if os.path.exists(img_path):
                try:
                    # Load image and compute encoding
                    image = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_encodings.append(encodings[0])
                        self.known_metadata.append({
                            "name": person["name"],
                            "identifier": person["identifier"],
                            "status": person["status"]
                        })
                        print(f"Loaded face encoding for: {person['name']} ({person['status']})")
                    else:
                        print(f"Warning: No face found in reference photo for {person['name']}")
                except Exception as e:
                    print(f"Error loading face encoding for {person['name']}: {e}")
            else:
                print(f"Warning: Reference image not found for {person['name']} at {img_path}")

    def register_new_face(self, name, identifier, status, frame):
        # frame: numpy array (OpenCV image) of the person
        if face_recognition is None:
            print("Error: face_recognition is not available.")
            return False, "Librería face_recognition no disponible."

        # Detect faces in the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if len(face_locations) == 0:
            return False, "No se detectó ningún rostro en la imagen."
        if len(face_locations) > 1:
            return False, "Se detectó más de un rostro. Tome la foto a una sola persona."

        # Compute encoding to verify it compiles correctly
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        if not encodings:
            return False, "No se pudo codificar el rostro."

        # Save reference image
        os.makedirs("data/known_faces", exist_ok=True)
        img_path = f"data/known_faces/{identifier}.jpg"
        
        # Crop the face or save the whole image?
        # Saving the whole frame is usually safer, but we can also save a tight crop
        # Let's save the frame where they look at the camera
        cv2.imwrite(img_path, frame)
        
        # Save to database
        try:
            self.db.register_person(name, identifier, status, img_path)
            
            # Update cache in memory
            self.known_encodings.append(encodings[0])
            self.known_metadata.append({
                "name": name,
                "identifier": identifier,
                "status": status
            })
            return True, "Registro exitoso."
        except Exception as e:
            return False, f"Error al guardar en base de datos: {e}"

    def get_known_faces_cache(self):
        return self.known_encodings, self.known_metadata
