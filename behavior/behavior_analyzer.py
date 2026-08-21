import time
from datetime import datetime

class BehaviorAnalyzer:
    def __init__(self, config):
        self.config = config
        
        # Track fall frames to avoid momentary glitches
        # object_id -> count of consecutive frames with fall aspect ratio
        # Track fall frames to avoid momentary glitches
        # object_id -> count of consecutive frames with fall aspect ratio
        self.fall_counters = {}
        # Track history of Y-centroids and height: object_id -> list of (cy, height)
        self.centroid_history = {}

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

        # 3. Calibrated Fall Detection
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        aspect_ratio = w / h if h > 0 else 0.0
        cy = int((y1 + y2) / 2.0)
        
        if object_id not in self.centroid_history:
            self.centroid_history[object_id] = []
        self.centroid_history[object_id].append((cy, h))
        if len(self.centroid_history[object_id]) > 15:
            self.centroid_history[object_id].pop(0)
            
        # Fall requirements:
        # A. Bounding box shape is flat / horizontal (aspect_ratio >= 1.35)
        is_flat = aspect_ratio >= 1.35
        
        # B. Bounding box height collapsed or centroid dropped rapidly
        has_dropped = False
        if len(self.centroid_history[object_id]) >= 5:
            oldest_cy, oldest_h = self.centroid_history[object_id][0]
            # cy increases downwards in pixel space; check for a drop (cy increase) or collapse in height
            if (cy - oldest_cy) > int(oldest_h * 0.15) or h < int(oldest_h * 0.65):
                has_dropped = True
                
        if is_flat and has_dropped:
            self.fall_counters[object_id] = self.fall_counters.get(object_id, 0) + 1
            if self.fall_counters[object_id] >= 8: # must persist for 8 frames
                violations.append({
                    "type": "CAIDA_DETECTADA",
                    "description": "Caída de persona detectada",
                    "zone": current_zones[0] if current_zones else "General",
                    "severity_score": 30
                })
        else:
            # Decay counter slowly to prevent flickering
            if object_id in self.fall_counters and self.fall_counters[object_id] > 0:
                self.fall_counters[object_id] -= 1
                
        return violations
        
    def cleanup_tracker(self, object_id):
        if object_id in self.fall_counters:
            del self.fall_counters[object_id]
        if object_id in self.centroid_history:
            del self.centroid_history[object_id]
