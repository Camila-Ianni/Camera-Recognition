import os
from dotenv import load_dotenv
from database.database import DatabaseManager

# Load environment variables
load_dotenv()

class AppConfig:
    def __init__(self, db_path=None):
        # Determine database path
        if not db_path:
            db_path = os.getenv("DB_PATH", "data/security_vision.db")
        self.db = DatabaseManager(db_path)
        
        # Initialize default configurations if not present in DB
        self._init_defaults()

    def _init_defaults(self):
        # WhatsApp settings
        if not self.db.get_setting("whatsapp_api_token"):
            self.db.set_setting("whatsapp_api_token", os.getenv("WHATSAPP_API_TOKEN", ""))
        if not self.db.get_setting("whatsapp_phone_number_id"):
            self.db.set_setting("whatsapp_phone_number_id", os.getenv("WHATSAPP_PHONE_NUMBER_ID", ""))
        if not self.db.get_setting("whatsapp_recipient_phone"):
            self.db.set_setting("whatsapp_recipient_phone", os.getenv("WHATSAPP_RECIPIENT_PHONE", ""))
            
        # Analysis toggles
        if not self.db.get_setting("enable_face_recognition"):
            self.db.set_setting("enable_face_recognition", "True")
        if not self.db.get_setting("enable_emotion_detection"):
            self.db.set_setting("enable_emotion_detection", "True")
        if not self.db.get_setting("enable_object_detection"):
            self.db.set_setting("enable_object_detection", "True")
        if not self.db.get_setting("enable_whatsapp_alerts"):
            self.db.set_setting("enable_whatsapp_alerts", "False")
            
        # Camera index / Video simulation path
        if not self.db.get_setting("camera_index"):
            self.db.set_setting("camera_index", os.getenv("DEFAULT_CAMERA_INDEX", "0"))
            
        # Detection properties
        if not self.db.get_setting("min_confidence"):
            self.db.set_setting("min_confidence", os.getenv("DEFAULT_MIN_CONFIDENCE", "0.5"))
        if not self.db.get_setting("risk_objects"):
            self.db.set_setting("risk_objects", os.getenv("DEFAULT_RISK_OBJECTS", "knife,scissors,backpack,handbag,suitcase,cell phone"))
            
        # Behavior guidelines
        if not self.db.get_setting("restricted_hours_start"):
            self.db.set_setting("restricted_hours_start", os.getenv("RESTRICTED_HOURS_START", "22"))
        if not self.db.get_setting("restricted_hours_end"):
            self.db.set_setting("restricted_hours_end", os.getenv("RESTRICTED_HOURS_END", "6"))
        if not self.db.get_setting("restricted_dwell_time"):
            self.db.set_setting("restricted_dwell_time", "10") # in seconds
            
        # Score risk thresholds
        if not self.db.get_setting("risk_threshold_medium"):
            self.db.set_setting("risk_threshold_medium", "25")
        if not self.db.get_setting("risk_threshold_critical"):
            self.db.set_setting("risk_threshold_critical", "50")

    # Helper getters/setters that convert data types
    def get_whatsapp_config(self):
        return {
            "token": self.db.get_setting("whatsapp_api_token", ""),
            "phone_id": self.db.get_setting("whatsapp_phone_number_id", ""),
            "recipient": self.db.get_setting("whatsapp_recipient_phone", "")
        }

    def set_whatsapp_config(self, token, phone_id, recipient):
        self.db.set_setting("whatsapp_api_token", token)
        self.db.set_setting("whatsapp_phone_number_id", phone_id)
        self.db.set_setting("whatsapp_recipient_phone", recipient)

    def get_camera_source(self):
        source = self.db.get_setting("camera_index", "0")
        # If it's a digit, return as integer (webcam index)
        if source.isdigit():
            return int(source)
        return source # Return string (video file path)

    def set_camera_source(self, source):
        self.db.set_setting("camera_index", str(source))

    def get_min_confidence(self):
        return float(self.db.get_setting("min_confidence", 0.5))

    def set_min_confidence(self, val):
        self.db.set_setting("min_confidence", str(val))

    def get_risk_objects(self):
        raw = self.db.get_setting("risk_objects", "")
        return [x.strip() for x in raw.split(",") if x.strip()]

    def set_risk_objects(self, lst):
        if isinstance(lst, list):
            self.db.set_setting("risk_objects", ",".join(lst))
        else:
            self.db.set_setting("risk_objects", str(lst))

    def get_restricted_hours(self):
        return (
            int(self.db.get_setting("restricted_hours_start", 22)),
            int(self.db.get_setting("restricted_hours_end", 6))
        )

    def set_restricted_hours(self, start, end):
        self.db.set_setting("restricted_hours_start", str(start))
        self.db.set_setting("restricted_hours_end", str(end))

    def get_dwell_time_threshold(self):
        return int(self.db.get_setting("restricted_dwell_time", 10))

    def set_dwell_time_threshold(self, val):
        self.db.set_setting("restricted_dwell_time", str(val))

    def get_risk_thresholds(self):
        return (
            int(self.db.get_setting("risk_threshold_medium", 25)),
            int(self.db.get_setting("risk_threshold_critical", 50))
        )

    def set_risk_thresholds(self, medium, critical):
        self.db.set_setting("risk_threshold_medium", str(medium))
        self.db.set_setting("risk_threshold_critical", str(critical))

    def is_module_enabled(self, module_name):
        setting_key = f"enable_{module_name}"
        return self.db.get_setting(setting_key, "True") == "True"

    def set_module_enabled(self, module_name, enabled):
        setting_key = f"enable_{module_name}"
        self.db.set_setting(setting_key, "True" if enabled else "False")
