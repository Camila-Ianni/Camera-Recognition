import time
from datetime import datetime

class AlertManager:
    def __init__(self, config, database, evidence_manager, whatsapp_client):
        self.config = config
        self.db = database
        self.evidence = evidence_manager
        self.whatsapp = whatsapp_client
        
        # Cooldown management: (tracker_id, alert_type) -> timestamp of last alert
        # Prevents spamming alerts for the same event
        self.alert_cooldowns = {}
        self.cooldown_duration = 30.0 # seconds

    def update_evidence_path_in_db(self, event_id, video_path):
        # Callback for EvidenceManager once video compilation is done
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE events SET evidence_path = ? WHERE id = ?",
                (video_path, event_id)
            )
            conn.commit()
        print(f"Database event {event_id} updated with evidence video: {video_path}")

    def evaluate_and_alert(self, tracker_id, bbox, name, status, emotions, detected_objects, 
                           zones, behavior_violations, camera_manager):
        # Calculate Risk Score
        # Start at 0
        risk_score = 0
        reasons = []
        primary_object = None
        primary_confidence = 0.5
        
        # 1. Unknown Person (+10)
        if status == "unknown":
            risk_score += 10
            reasons.append("Persona desconocida")
            primary_confidence = 0.8
            
        # 2. Unauthorized registered person (+30)
        elif status == "unauthorized":
            risk_score += 30
            reasons.append(f"Persona no autorizada: {name}")
            primary_confidence = 0.95
            
        # 3. Behavior violations (e.g. dwell time +20, off-hours +20, fall +30)
        for violation in behavior_violations:
            risk_score += violation["severity_score"]
            reasons.append(violation["description"])
            
        # 4. Check for risk objects carried by or near the person
        # In a frame, if there is a risk object in the same zone, or overall, add score
        risk_objects_config = self.config.get_risk_objects()
        
        for obj in detected_objects:
            obj_name = obj["class_name"]
            if obj_name in risk_objects_config:
                risk_score += 50
                reasons.append(f"Objeto de riesgo detectado: {obj_name}")
                primary_object = obj_name
                primary_confidence = obj["confidence"]
                
        if not reasons:
            return None # No risk detected

        # Classify alert levels
        med_threshold, crit_threshold = self.config.get_risk_thresholds()
        
        if risk_score >= crit_threshold:
            alert_level = "CRÍTICO"
        elif risk_score >= med_threshold:
            alert_level = "ADVERTENCIA"
        else:
            alert_level = "INFORMACIÓN"

        # Check cooldown
        current_time = time.time()
        alert_type = reasons[-1] # Use the most recent/severe reason as trigger type
        cooldown_key = (tracker_id, alert_type)
        
        if cooldown_key in self.alert_cooldowns:
            if current_time - self.alert_cooldowns[cooldown_key] < self.cooldown_duration:
                # Suppressed due to cooldown, but return the active alert state for overlay
                return {
                    "level": alert_level,
                    "score": risk_score,
                    "reasons": reasons,
                    "suppressed": True
                }

        # Update cooldown
        self.alert_cooldowns[cooldown_key] = current_time

        # Format details for logging
        event_type = ", ".join(reasons)
        emotion_str = None
        if emotions:
            # Sort emotions and get top one
            top_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_str = f"{top_emotion[0]} ({int(top_emotion[1])}%)"
            
        zone_str = zones[0] if zones else "General"
        
        # Log to Database
        event_id = self.db.log_event(
            event_type=event_type,
            person_name=name if status != "unknown" else "Desconocido",
            object_name=primary_object,
            emotion=emotion_str,
            risk_score=risk_score,
            alert_level=alert_level,
            confidence=primary_confidence,
            zone=zone_str,
            evidence_path=None # Filled asynchronously by EvidenceManager
        )
        
        # Trigger Evidence capture (video recording)
        self.evidence.trigger_evidence_capture(
            camera_manager=camera_manager,
            event_id=event_id,
            event_name=alert_type,
            completion_callback=self.update_evidence_path_in_db
        )

        # Trigger WhatsApp if Critical or Warning
        if alert_level in ["CRÍTICO", "ADVERTENCIA"]:
            self.whatsapp.send_alert_async(
                event_type=event_type,
                person_name=name if status != "unknown" else "Desconocida",
                object_name=primary_object,
                zone=zone_str,
                level=alert_level,
                confidence=primary_confidence
            )

        return {
            "level": alert_level,
            "score": risk_score,
            "reasons": reasons,
            "suppressed": False
        }
        
    def clean_tracker_cooldowns(self, tracker_id):
        # Remove cooldowns for a tracker when it is deregistered
        for key in list(self.alert_cooldowns.keys()):
            if key[0] == tracker_id:
                del self.alert_cooldowns[key]
