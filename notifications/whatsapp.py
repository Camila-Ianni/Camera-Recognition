import requests
import threading
import os
from datetime import datetime

class WhatsAppClient:
    def __init__(self, config):
        self.config = config

    def send_alert_async(self, event_type, person_name, object_name, zone, level, confidence, timestamp=None):
        if not self.config.is_module_enabled("whatsapp_alerts"):
            print("WhatsApp alerts are disabled in configuration.")
            return
            
        # Start a thread to send the message asynchronously
        thread = threading.Thread(
            target=self._send_alert_worker,
            args=(event_type, person_name, object_name, zone, level, confidence, timestamp),
            daemon=True
        )
        thread.start()

    def _send_alert_worker(self, event_type, person_name, object_name, zone, level, confidence, timestamp):
        if not timestamp:
            timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            
        whatsapp_config = self.config.get_whatsapp_config()
        token = whatsapp_config.get("token", "")
        phone_id = whatsapp_config.get("phone_id", "")
        recipient = whatsapp_config.get("recipient", "")

        # Format message content
        msg_body = (
            f"🔥 AI SECURITY VISION — ALERTA 🔥\n\n"
            f"Tipo:\n{event_type}\n\n"
            f"Fecha y Hora:\n{timestamp}\n\n"
            f"Persona:\n{person_name if person_name else 'Desconocida'}\n\n"
            f"Objeto:\n{object_name if object_name else 'Ninguno'}\n\n"
            f"Zona:\n{zone}\n\n"
            f"Nivel:\n{level}\n\n"
            f"Confianza:\n{int(confidence * 100)}%"
        )

        # Check if we have complete configurations
        if not token or not phone_id or not recipient or "your_" in token or "your_" in phone_id:
            # Mock mode
            print("--------------------------------------------------")
            print("🔔 WHATSAPP API MOCK MODE")
            print(f"Recipient: {recipient}")
            print(f"Message:\n{msg_body}")
            print("--------------------------------------------------")
            
            # Log mock dispatch to file
            try:
                with open("logs/whatsapp_mock.log", "a") as f:
                    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Recipient: {recipient}\n{msg_body}\n\n")
            except Exception as e:
                print(f"Error logging mock whatsapp message: {e}")
            return

        # Real API dispatch
        url = f"https://graph.facebook.com/v18.0/{phone_id}/messages"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "text",
            "text": {
                "body": msg_body
            }
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            if response.status_code in [200, 201]:
                print(f"WhatsApp alert sent successfully to {recipient}.")
            else:
                print(f"WhatsApp API Error (Status {response.status_code}): {response.text}")
                # Log error
                with open("logs/whatsapp_errors.log", "a") as f:
                    f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {response.status_code} - {response.text}\n")
        except Exception as e:
            print(f"Error sending WhatsApp request: {e}")
            with open("logs/whatsapp_errors.log", "a") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Request Exception: {str(e)}\n")
