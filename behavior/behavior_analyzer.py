import time
from datetime import datetime

class BehaviorAnalyzer:
    def __init__(self, config):
        self.config = config
        
        # Track fall frames to avoid momentary glitches
        # object_id -> count of consecutive frames with fall aspect ratio
        self.fall_counters = {}

    def is_outside_permitted_hours(self):
        start_hour, end_hour = self.config.get_restricted_hours()
        current_hour = datetime.now().hour
        
        if start_hour == end_hour:
            return False # Permitted all hours
            
        if start_hour > end_hour:
            # Over-midnight range (e.g., 22:00 to 06:00)
            return current_hour >= start_hour or current_hour < end_hour
        else:
            # Single-day range (e.g., 01:00 to 05:00)
            return start_hour <= current_hour < end_hour

    def analyze_person(self, object_id, bbox, duration, current_zones, zone_manager):
        # bbox: (x1, y1, x2, y2)
        # duration: time in seconds since registered
        # current_zones: list of zone names the person is in
        
        violations = []
        
        # 1. Check Restricted Zone Dwell Time
        dwell_limit = self.config.get_dwell_time_threshold()
        in_restricted_zone = False
        
        for zone_name in current_zones:
            zone_info = zone_manager.get_zones().get(zone_name)
            if zone_info and zone_info.get("restricted", False):
                in_restricted_zone = True
                if duration >= dwell_limit:
                    violations.append({
                        "type": "PERMANENCIA_PROLONGADA",
                        "description": f"Permanencia prolongada ({int(duration)}s) en {zone_name}",
                        "zone": zone_name,
                        "severity_score": 20
                    })
                    
        # 2. Check Off-hours/Restricted hours
        if self.is_outside_permitted_hours():
            violations.append({
                "type": "FUERA_HORARIO",
                "description": "Presencia de persona fuera del horario permitido",
                "zone": current_zones[0] if current_zones else "General",
                "severity_score": 20
            })

        # 3. Fall Detection
        # Bounding box coordinates
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        if h > 0:
            aspect_ratio = w / h
            # A standing person typically has aspect_ratio around 0.3 - 0.5.
            # A fallen/lying down person has aspect_ratio > 1.0 (width > height).
            if aspect_ratio >= 1.2:
                self.fall_counters[object_id] = self.fall_counters.get(object_id, 0) + 1
                if self.fall_counters[object_id] >= 10: # must persist for 10 frames
                    violations.append({
                        "type": "CAIDA_DETECTADA",
                        "description": "Caída de persona detectada",
                        "zone": current_zones[0] if current_zones else "General",
                        "severity_score": 30
                    })
            else:
                self.fall_counters[object_id] = 0
                
        return violations
        
    def cleanup_tracker(self, object_id):
        if object_id in self.fall_counters:
            del self.fall_counters[object_id]
