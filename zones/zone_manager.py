import json
import cv2
import numpy as np

class ZoneManager:
    def __init__(self, db_manager=None):
        self.db = db_manager
        self.zones = {} # name -> {"points": [(nx, ny), ...], "restricted": bool}
        
        if self.db:
            self.load_zones()
        else:
            # Set default zones if no DB is connected
            self.zones = {
                "Entrada": {
                    "points": [(0.0, 0.0), (0.4, 0.0), (0.4, 1.0), (0.0, 1.0)],
                    "restricted": False
                },
                "Zona Restringida": {
                    "points": [(0.6, 0.2), (0.95, 0.2), (0.95, 0.8), (0.6, 0.8)],
                    "restricted": True
                }
            }

    def add_zone(self, name, points, is_restricted=False):
        # points should be list of (nx, ny) normalized coordinates (0.0 to 1.0)
        self.zones[name] = {
            "points": points,
            "restricted": is_restricted
        }
        if self.db:
            self.save_zones()

    def remove_zone(self, name):
        if name in self.zones:
            del self.zones[name]
            if self.db:
                self.save_zones()
            return True
        return False

    def get_zones(self):
        return self.zones

    def save_zones(self):
        if not self.db:
            return
        serialized = json.dumps(self.zones)
        self.db.set_setting("safety_zones", serialized)

    def load_zones(self):
        if not self.db:
            return
        serialized = self.db.get_setting("safety_zones")
        if serialized:
            try:
                self.zones = json.loads(serialized)
            except Exception as e:
                print(f"Error loading zones from DB: {e}")
                self.zones = {}
        else:
            # Set defaults
            self.zones = {
                "Entrada": {
                    "points": [(0.0, 0.1), (0.4, 0.1), (0.4, 0.9), (0.0, 0.9)],
                    "restricted": False
                },
                "Zona Restringida": {
                    "points": [(0.6, 0.1), (0.95, 0.1), (0.95, 0.9), (0.6, 0.9)],
                    "restricted": True
                }
            }
            self.save_zones()

    def check_point_in_zone(self, px, py, width, height, zone_name):
        if zone_name not in self.zones:
            return False
            
        pts = self.zones[zone_name]["points"]
        if not pts:
            return False
            
        # Convert normalized coordinates to absolute pixel coordinates
        pixel_points = []
        for nx, ny in pts:
            pixel_points.append([int(nx * width), int(ny * height)])
            
        contour = np.array(pixel_points, dtype=np.int32).reshape((-1, 1, 2))
        
        # cv2.pointPolygonTest returns:
        # +1 if inside, 0 if on edge, -1 if outside
        dist = cv2.pointPolygonTest(contour, (float(px), float(py)), False)
        return dist >= 0

    def get_zones_for_point(self, px, py, width, height):
        inside_zones = []
        for zone_name in self.zones:
            if self.check_point_in_zone(px, py, width, height, zone_name):
                inside_zones.append(zone_name)
        return inside_zones
