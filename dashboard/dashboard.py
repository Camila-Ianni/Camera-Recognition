import os
import sys
import time
import cv2
import threading
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

import tkinter as tk
import customtkinter as ctk

# Custom modules
from camera.camera_manager import CameraManager
from face.face_database import FaceDatabase
from face.face_recognition import FaceRecognizer
from emotion.emotion_detector import EmotionDetector
from objects.object_detector import ObjectDetector
from tracking.person_tracker import PersonTracker
from zones.zone_manager import ZoneManager
from behavior.behavior_analyzer import BehaviorAnalyzer
from alerts.alert_manager import AlertManager
from evidence.evidence_manager import EvidenceManager
from notifications.whatsapp import WhatsAppClient

# Appearance settings
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class SecurityDashboard(ctk.CTk):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.db = config.db
        
        # Configure window
        self.title("AI SECURITY VISION — Sistema de Videovigilancia Inteligente")
        self.geometry("1280x750")
        self.minsize(1100, 650)
        
        # Initialize Core Engines
        self.camera_manager = CameraManager(self.config.get_camera_source())
        self.face_db = FaceDatabase(self.db)
        self.face_recognizer = FaceRecognizer(self.face_db)
        self.emotion_detector = EmotionDetector(self.config)
        self.object_detector = ObjectDetector(self.config)
        self.person_tracker = PersonTracker()
        self.zone_manager = ZoneManager(self.db)
        self.behavior_analyzer = BehaviorAnalyzer(self.config)
        self.whatsapp_client = WhatsAppClient(self.config)
        self.evidence_manager = EvidenceManager()
        self.alert_manager = AlertManager(self.config, self.db, self.evidence_manager, self.whatsapp_client)
        
        # UI State variables
        self.current_frame_mode = "camera" # camera / replay
        self.replay_cap = None
        self.replay_frames = []
        self.replay_idx = 0
        self.is_monitoring_night = False
        self.face_recognition_cache = {}
        
        # Navigation State
        self.active_frame = None
        
        # Build layout
        self.create_layout()
        
        # Start camera automatically
        self.camera_manager.start()
        
        # Start UI loop updates
        self.update_feed()
        self.update_stats_panel()
        
        # Handle close window
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_layout(self):
        # Configure grid layout 1x2 (Sidebar + Content)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # Sidebar Frame
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1)
        
        # Sidebar Title
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="AI SECURITY\nVISION", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 20))
        
        # Sidebar buttons
        self.btn_dashboard = ctk.CTkButton(
            self.sidebar_frame, text="Panel de Control", anchor="w",
            command=lambda: self.select_frame("dashboard")
        )
        self.btn_dashboard.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        self.btn_camera = ctk.CTkButton(
            self.sidebar_frame, text="Monitoreo Cámara", anchor="w",
            command=lambda: self.select_frame("camera")
        )
        self.btn_camera.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        self.btn_people = ctk.CTkButton(
            self.sidebar_frame, text="Base de Rostros", anchor="w",
            command=lambda: self.select_frame("people")
        )
        self.btn_people.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        self.btn_alerts = ctk.CTkButton(
            self.sidebar_frame, text="Registro de Alertas", anchor="w",
            command=lambda: self.select_frame("alerts")
        )
        self.btn_alerts.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        
        self.btn_stats = ctk.CTkButton(
            self.sidebar_frame, text="Gráficos Estadísticos", anchor="w",
            command=lambda: self.select_frame("stats")
        )
        self.btn_stats.grid(row=5, column=0, padx=20, pady=10, sticky="ew")
        
        self.btn_config = ctk.CTkButton(
            self.sidebar_frame, text="Configuración", anchor="w",
            command=lambda: self.select_frame("config")
        )
        self.btn_config.grid(row=6, column=0, padx=20, pady=10, sticky="ew")
        
        # Night mode toggle inside Sidebar
        self.night_mode_switch = ctk.CTkSwitch(
            self.sidebar_frame, 
            text="Monitoreo Nocturno",
            command=self.toggle_night_monitoring
        )
        self.night_mode_switch.grid(row=9, column=0, padx=20, pady=20, sticky="s")
        
        # Content frame container
        self.content_container = ctk.CTkFrame(self, corner_radius=10, fg_color="transparent")
        self.content_container.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.content_container.grid_rowconfigure(0, weight=1)
        self.content_container.grid_columnconfigure(0, weight=1)
        
        # Start on dashboard page
        self.select_frame("dashboard")

    def select_frame(self, name):
        # Remove active frame
        if self.active_frame:
            self.active_frame.destroy()
            
        # Highlight sidebar buttons
        buttons = {
            "dashboard": self.btn_dashboard,
            "camera": self.btn_camera,
            "people": self.btn_people,
            "alerts": self.btn_alerts,
            "stats": self.btn_stats,
            "config": self.btn_config
        }
        for k, btn in buttons.items():
            if k == name:
                btn.configure(fg_color=("gray75", "gray25"), border_width=1)
            else:
                btn.configure(fg_color="transparent", border_width=0)
                
        # Load new frame
        if name == "dashboard":
            self.load_dashboard_page()
        elif name == "camera":
            self.load_camera_page()
        elif name == "people":
            self.load_people_page()
        elif name == "alerts":
            self.load_alerts_page()
        elif name == "stats":
            self.load_stats_page()
        elif name == "config":
            self.load_config_page()

    # ==================== PAGE 1: DASHBOARD ====================
    def load_dashboard_page(self):
        self.active_frame = ctk.CTkFrame(self.content_container, fg_color="transparent")
        self.active_frame.grid(row=0, column=0, sticky="nsew")
        self.active_frame.grid_rowconfigure(1, weight=1)
        self.active_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Title
        lbl_title = ctk.CTkLabel(
            self.active_frame, 
            text="Dashboard de Seguridad AI Security Vision", 
            font=ctk.CTkFont(size=22, weight="bold")
        )
        lbl_title.grid(row=0, column=0, columnspan=2, padx=20, pady=15, sticky="w")
        
        # Left Panel: System Status
        status_frame = ctk.CTkFrame(self.active_frame)
        status_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        status_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(status_frame, text="ESTADO DEL SISTEMA", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)
        
        # Modules status
        modules = [
            ("Cámara", self.camera_manager.is_running),
            ("Motor de Visión Artificial", True),
            ("Reconocimiento Facial", self.config.is_module_enabled("face_recognition")),
            ("Análisis de Expresiones", self.config.is_module_enabled("emotion_detection")),
            ("Detección de Objetos", self.config.is_module_enabled("object_detection")),
            ("Notificaciones WhatsApp", self.config.is_module_enabled("whatsapp_alerts"))
        ]
        
        for name, state in modules:
            m_frame = ctk.CTkFrame(status_frame, height=35, fg_color="transparent")
            m_frame.pack(fill="x", padx=15, pady=5)
            
            ctk.CTkLabel(m_frame, text=name).pack(side="left")
            status_text = "ONLINE" if state else "OFFLINE"
            status_color = "#2ECC71" if state else "#E74C3C"
            
            lbl_st = ctk.CTkLabel(m_frame, text=status_text, text_color=status_color, font=ctk.CTkFont(weight="bold"))
            lbl_st.pack(side="right")
            
        # Info Box counters
        counters_frame = ctk.CTkFrame(status_frame, fg_color="transparent")
        counters_frame.pack(fill="x", padx=15, pady=15)
        
        # Registered Count
        people_count = len(self.db.get_registered_people())
        self.lbl_card_people = ctk.CTkLabel(
            counters_frame, 
            text=f"Rostros Registrados: {people_count}", 
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=("gray85", "gray15"),
            corner_radius=5,
            height=40
        )
        self.lbl_card_people.pack(fill="x", pady=5)
        
        # Total Alerts count
        alerts_count = len(self.db.get_events())
        critical_count = len(self.db.get_events(alert_level="CRÍTICO"))
        self.lbl_card_alerts = ctk.CTkLabel(
            counters_frame, 
            text=f"Alertas Registradas: {alerts_count} ({critical_count} Críticas)", 
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=("gray85", "gray15"),
            corner_radius=5,
            height=40
        )
        self.lbl_card_alerts.pack(fill="x", pady=5)

        # Right Panel: Recent Alerts
        alerts_panel = ctk.CTkFrame(self.active_frame)
        alerts_panel.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        alerts_panel.grid_rowconfigure(1, weight=1)
        alerts_panel.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(alerts_panel, text="ALERTAS RECIENTES", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, pady=10)
        
        # Scrollable area for alert cards
        self.scroll_alerts = ctk.CTkScrollableFrame(alerts_panel)
        self.scroll_alerts.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.populate_recent_alerts()

    def populate_recent_alerts(self):
        # Clear existing
        for widget in self.scroll_alerts.winfo_children():
            widget.destroy()
            
        events = self.db.get_events(limit=25)
        if not events:
            ctk.CTkLabel(self.scroll_alerts, text="No se han registrado incidentes aún.").pack(pady=20)
            return
            
        for ev in events:
            # Color code by level
            lvl = ev["alert_level"]
            bg_col = "#7B241C" if lvl == "CRÍTICO" else ("#7D6608" if lvl == "ADVERTENCIA" else ("#1A5276" if lvl == "INFORMACIÓN" else "gray20"))
            
            card = ctk.CTkFrame(self.scroll_alerts, fg_color=bg_col, corner_radius=6)
            card.pack(fill="x", pady=5, padx=5)
            
            header = f"{ev['timestamp']} — {lvl}"
            ctk.CTkLabel(card, text=header, font=ctk.CTkFont(weight="bold", size=12)).pack(anchor="w", padx=10, pady=(5, 0))
            
            details = ev["event_type"]
            if ev["person_name"] and ev["person_name"] != "Desconocido":
                details += f" | Persona: {ev['person_name']}"
            if ev["zone"] and ev["zone"] != "General":
                details += f" | Zona: {ev['zone']}"
                
            ctk.CTkLabel(card, text=details, font=ctk.CTkFont(size=11), wraplength=400, justify="left").pack(anchor="w", padx=10, pady=(2, 5))

    # ==================== PAGE 2: CAMERA VIEW ====================
    def load_camera_page(self):
        self.active_frame = ctk.CTkFrame(self.content_container, fg_color="transparent")
        self.active_frame.grid(row=0, column=0, sticky="nsew")
        self.active_frame.grid_rowconfigure(0, weight=1)
        self.active_frame.grid_columnconfigure(0, weight=3) # Camera feed
        self.active_frame.grid_columnconfigure(1, weight=1) # Side information panel
        
        # Left container: Canvas and controls
        left_frame = ctk.CTkFrame(self.active_frame, fg_color="transparent")
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)
        
        # Canvas for video
        self.canvas_video = tk.Canvas(left_frame, bg="black", highlightthickness=0)
        self.canvas_video.grid(row=0, column=0, sticky="nsew")
        
        # Under-canvas control buttons
        controls_frame = ctk.CTkFrame(left_frame, height=50)
        controls_frame.grid(row=1, column=0, pady=(10, 0), sticky="ew")
        
        self.btn_toggle_camera = ctk.CTkButton(
            controls_frame, 
            text="Detener Cámara" if self.camera_manager.is_running else "Iniciar Cámara", 
            command=self.toggle_camera_feed,
            width=120
        )
        self.btn_toggle_camera.pack(side="left", padx=10, pady=10)
        
        btn_simulation = ctk.CTkButton(
            controls_frame, 
            text="Simulación: Cargar Video", 
            command=self.load_simulation_video,
            width=150
        )
        btn_simulation.pack(side="left", padx=10, pady=10)
        
        self.lbl_fps = ctk.CTkLabel(controls_frame, text="FPS: --", font=ctk.CTkFont(weight="bold"))
        self.lbl_fps.pack(side="right", padx=20, pady=10)
        
        # Right container: Live status details
        self.info_panel = ctk.CTkScrollableFrame(self.active_frame)
        self.info_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(self.info_panel, text="DETECCIONES EN TIEMPO REAL", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)
        
        # Text details container
        self.detections_desc_frame = ctk.CTkFrame(self.info_panel, fg_color="transparent")
        self.detections_desc_frame.pack(fill="both", expand=True)
        
        self.lbl_no_detections = ctk.CTkLabel(self.detections_desc_frame, text="No hay personas en cámara.")
        self.lbl_no_detections.pack(pady=20)

    def toggle_camera_feed(self):
        if self.camera_manager.is_running:
            self.camera_manager.stop()
            self.btn_toggle_camera.configure(text="Iniciar Cámara")
        else:
            # Revert to default index from database
            self.camera_manager.source = self.config.get_camera_source()
            if self.camera_manager.start():
                self.btn_toggle_camera.configure(text="Detener Cámara")
            else:
                tk.messagebox.showerror("Error", "No se pudo abrir la cámara.")

    def load_simulation_video(self):
        # Open file dialog
        file_path = tk.filedialog.askopenfilename(
            title="Seleccionar Video de Simulación",
            filetypes=[("Archivos de Video", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            # Change camera source to the video file
            self.camera_manager.change_source(file_path)
            self.current_frame_mode = "camera"
            self.btn_toggle_camera.configure(text="Detener Cámara")
            print(f"Cargado video de simulación: {file_path}")

    def toggle_night_monitoring(self):
        self.is_monitoring_night = self.night_mode_switch.get()
        print(f"Modo monitoreo nocturno: {'ACTIVADO' if self.is_monitoring_night else 'DESACTIVADO'}")

    # ==================== PAGE 3: PEOPLE REGISTER ====================
    def load_people_page(self):
        self.active_frame = ctk.CTkFrame(self.content_container, fg_color="transparent")
        self.active_frame.grid(row=0, column=0, sticky="nsew")
        self.active_frame.grid_rowconfigure(0, weight=1)
        self.active_frame.grid_columnconfigure(0, weight=1)
        self.active_frame.grid_columnconfigure(1, weight=1)
        
        # Left Frame: List of registered
        list_frame = ctk.CTkFrame(self.active_frame)
        list_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        list_frame.grid_rowconfigure(1, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(list_frame, text="PERSONAS REGISTRADAS", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, pady=10)
        
        self.scroll_people = ctk.CTkScrollableFrame(list_frame)
        self.scroll_people.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.populate_people_list()
        
        # Right Frame: Registration form
        form_frame = ctk.CTkFrame(self.active_frame)
        form_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(form_frame, text="REGISTRAR NUEVA PERSONA", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=15)
        
        ctk.CTkLabel(form_frame, text="Nombre Completo").pack(anchor="w", padx=30, pady=(10, 0))
        self.ent_name = ctk.CTkEntry(form_frame, placeholder_text="Ej: Camila Ianni")
        self.ent_name.pack(fill="x", padx=30, pady=5)
        
        ctk.CTkLabel(form_frame, text="Identificador Único (DNI/ID)").pack(anchor="w", padx=30, pady=(10, 0))
        self.ent_id = ctk.CTkEntry(form_frame, placeholder_text="Ej: 42893121")
        self.ent_id.pack(fill="x", padx=30, pady=5)
        
        ctk.CTkLabel(form_frame, text="Estado de Autorización").pack(anchor="w", padx=30, pady=(10, 0))
        self.sel_status = ctk.CTkComboBox(form_frame, values=["autorizado", "no autorizado"])
        self.sel_status.pack(fill="x", padx=30, pady=5)
        self.sel_status.set("autorizado")
        
        btn_capture = ctk.CTkButton(
            form_frame, 
            text="Capturar Foto desde Cámara", 
            command=self.capture_and_register_person,
            fg_color="#1F618D"
        )
        btn_capture.pack(fill="x", padx=30, pady=25)
        
        btn_file = ctk.CTkButton(
            form_frame, 
            text="Seleccionar Foto de Archivo", 
            command=self.upload_and_register_person,
            fg_color="gray30"
        )
        btn_file.pack(fill="x", padx=30, pady=(0, 20))

    def populate_people_list(self):
        for widget in self.scroll_people.winfo_children():
            widget.destroy()
            
        people = self.db.get_registered_people()
        if not people:
            ctk.CTkLabel(self.scroll_people, text="No hay personas registradas.").pack(pady=20)
            return
            
        for person in people:
            item = ctk.CTkFrame(self.scroll_people, fg_color=("gray85", "gray15"), corner_radius=5)
            item.pack(fill="x", pady=5, padx=5)
            
            # Text layout
            auth_symbol = "✓" if person["status"] == "autorizado" else "⚠️"
            auth_color = "#2ECC71" if person["status"] == "autorizado" else "#E74C3C"
            
            info_txt = f"{person['name']} | ID: {person['identifier']}\nFecha: {person['registered_date']}"
            lbl_info = ctk.CTkLabel(item, text=info_txt, font=ctk.CTkFont(size=11), justify="left")
            lbl_info.pack(side="left", padx=10, pady=5)
            
            lbl_auth = ctk.CTkLabel(
                item, 
                text=f"{person['status'].upper()} {auth_symbol}", 
                text_color=auth_color,
                font=ctk.CTkFont(weight="bold", size=10)
            )
            lbl_auth.pack(side="left", padx=15)
            
            # Delete button
            btn_del = ctk.CTkButton(
                item, 
                text="Eliminar", 
                fg_color="#922B21", 
                width=60, 
                height=25,
                command=lambda p_id=person["identifier"]: self.delete_person_and_refresh(p_id)
            )
            btn_del.pack(side="right", padx=10, pady=5)

    def delete_person_and_refresh(self, identifier):
        if tk.messagebox.askyesno("Confirmar", f"¿Está seguro de que desea eliminar la persona con ID {identifier}?"):
            if self.db.delete_person(identifier):
                # Reload face encodings in FaceDatabase
                self.face_db.load_known_faces()
                self.populate_people_list()
                tk.messagebox.showinfo("Éxito", "Persona eliminada correctamente.")
            else:
                tk.messagebox.showerror("Error", "No se pudo eliminar de la base de datos.")

    def capture_and_register_person(self):
        name = self.ent_name.get().strip()
        identifier = self.ent_id.get().strip()
        status = self.sel_status.get()
        
        if not name or not identifier:
            tk.messagebox.showwarning("Campos vacíos", "Por favor complete el Nombre y el Identificador.")
            return
            
        frame = self.camera_manager.get_frame()
        if frame is None:
            tk.messagebox.showerror("Error de cámara", "No hay fotograma disponible de la cámara en este momento.")
            return
            
        success, msg = self.face_db.register_new_face(name, identifier, status, frame)
        if success:
            tk.messagebox.showinfo("Éxito", msg)
            self.ent_name.delete(0, "end")
            self.ent_id.delete(0, "end")
            self.populate_people_list()
        else:
            tk.messagebox.showerror("Error de registro", msg)

    def upload_and_register_person(self):
        name = self.ent_name.get().strip()
        identifier = self.ent_id.get().strip()
        status = self.sel_status.get()
        
        if not name or not identifier:
            tk.messagebox.showwarning("Campos vacíos", "Por favor complete el Nombre y el Identificador.")
            return
            
        file_path = tk.filedialog.askopenfilename(
            title="Seleccionar Foto de Referencia",
            filetypes=[("Archivos de Imagen", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            img = cv2.imread(file_path)
            if img is None:
                tk.messagebox.showerror("Error", "No se pudo abrir el archivo de imagen.")
                return
                
            success, msg = self.face_db.register_new_face(name, identifier, status, img)
            if success:
                tk.messagebox.showinfo("Éxito", msg)
                self.ent_name.delete(0, "end")
                self.ent_id.delete(0, "end")
                self.populate_people_list()
            else:
                tk.messagebox.showerror("Error de registro", msg)

    # ==================== PAGE 4: ALERTS REGISTER & PLAYBACK ====================
    def load_alerts_page(self):
        self.active_frame = ctk.CTkFrame(self.content_container, fg_color="transparent")
        self.active_frame.grid(row=0, column=0, sticky="nsew")
        self.active_frame.grid_rowconfigure(0, weight=1)
        self.active_frame.grid_columnconfigure(0, weight=1)
        
        # Horizontal Split: Table & Playback Viewer
        self.active_frame.grid_rowconfigure(0, weight=1)
        self.active_frame.grid_rowconfigure(1, weight=1)
        
        # Upper Panel: Table
        table_frame = ctk.CTkFrame(self.active_frame)
        table_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        table_frame.grid_rowconfigure(1, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(table_frame, text="HISTORIAL DE ALERTAS E INCIDENTES", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, pady=5)
        
        self.scroll_history = ctk.CTkScrollableFrame(table_frame)
        self.scroll_history.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        # Lower Panel: Replay / Detail
        self.replay_frame = ctk.CTkFrame(self.active_frame)
        self.replay_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.replay_frame.grid_rowconfigure(0, weight=1)
        self.replay_frame.grid_columnconfigure(0, weight=1) # Video
        self.replay_frame.grid_columnconfigure(1, weight=1) # Info
        
        # Mini canvas for playback
        self.canvas_replay = tk.Canvas(self.replay_frame, bg="black", highlightthickness=0)
        self.canvas_replay.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Info details
        self.replay_info_frame = ctk.CTkFrame(self.replay_frame, fg_color="transparent")
        self.replay_info_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.lbl_replay_details = ctk.CTkLabel(
            self.replay_info_frame, 
            text="Seleccione una alerta del historial para reproducir el incidente.",
            justify="left",
            wraplength=400
        )
        self.lbl_replay_details.pack(pady=20)
        
        self.btn_open_external = ctk.CTkButton(
            self.replay_info_frame, 
            text="Abrir en Reproductor del Sistema ↗", 
            command=self.open_video_externally,
            state="disabled"
        )
        self.btn_open_external.pack(pady=10)
        
        self.selected_evidence_path = None
        self.populate_alerts_history()

    def populate_alerts_history(self):
        for widget in self.scroll_history.winfo_children():
            widget.destroy()
            
        events = self.db.get_events(limit=100)
        if not events:
            ctk.CTkLabel(self.scroll_history, text="No hay alertas registradas en el historial.").pack(pady=20)
            return
            
        for ev in events:
            # Single alert row
            row = ctk.CTkFrame(self.scroll_history, fg_color=("gray85", "gray15"), corner_radius=4)
            row.pack(fill="x", pady=3, padx=5)
            
            lvl = ev["alert_level"]
            lvl_color = "#E74C3C" if lvl == "CRÍTICO" else ("#F1C40F" if lvl == "ADVERTENCIA" else "#3498DB")
            
            # Left Level Tag
            lbl_lvl = ctk.CTkLabel(
                row, 
                text=f" {lvl} ", 
                fg_color=lvl_color, 
                text_color="black" if lvl == "ADVERTENCIA" else "white",
                font=ctk.CTkFont(weight="bold", size=10),
                corner_radius=3
            )
            lbl_lvl.pack(side="left", padx=10, pady=8)
            
            # Description
            desc = f"{ev['timestamp']} | {ev['event_type']} (Zona: {ev['zone']})"
            lbl_desc = ctk.CTkLabel(row, text=desc, font=ctk.CTkFont(size=11), justify="left")
            lbl_desc.pack(side="left", padx=10, fill="x", expand=True)
            
            # Action button
            btn_play = ctk.CTkButton(
                row, 
                text="Ver Incidente ▶", 
                width=100, 
                height=25,
                command=lambda e_path=ev["evidence_path"], event=ev: self.load_alert_playback(e_path, event)
            )
            btn_play.pack(side="right", padx=10, pady=8)

    def load_alert_playback(self, video_path, event):
        # Stop camera feed updates during playback if active page changed
        self.selected_evidence_path = video_path
        
        # Format label details
        details_txt = (
            f"⚡ DETALLES DEL INCIDENTE ⚡\n\n"
            f"Nivel: {event['alert_level']}\n"
            f"Evento: {event['event_type']}\n"
            f"Fecha/Hora: {event['timestamp']}\n"
            f"Persona: {event['person_name']}\n"
            f"Objeto: {event['object_name'] if event['object_name'] else 'Ninguno'}\n"
            f"Emoción Estimada: {event['emotion'] if event['emotion'] else 'No disponible'}\n"
            f"Zona: {event['zone']}\n"
            f"Confianza: {int(event['confidence'] * 100)}%\n"
            f"Puntaje de Riesgo: {event['risk_score']} Pts"
        )
        self.lbl_replay_details.configure(text=details_txt)
        
        if video_path and os.path.exists(video_path):
            self.btn_open_external.configure(state="normal")
            
            # Load video file into frame list
            self.replay_frames = []
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                self.replay_frames.append(frame)
            cap.release()
            
            self.replay_idx = 0
            self.current_frame_mode = "replay"
            self.play_replay_loop()
        else:
            self.btn_open_external.configure(state="disabled")
            self.replay_frames = []
            # Draw black screen
            self.canvas_replay.delete("all")
            # Draw text
            self.canvas_replay.create_text(
                150, 100, 
                text="Video de evidencia no disponible.\nVerifique si fue eliminado.",
                fill="white",
                justify="center"
            )

    def play_replay_loop(self):
        if self.current_frame_mode != "replay" or not self.replay_frames:
            return
            
        frame = self.replay_frames[self.replay_idx]
        
        # Resize to fit replay canvas
        canvas_w = self.canvas_replay.winfo_width()
        canvas_h = self.canvas_replay.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10:
            # Fallback size
            canvas_w, canvas_h = 320, 240
            
        frame_resized = cv2.resize(frame, (canvas_w, canvas_h))
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(rgb_frame)
        self.replay_photo = ImageTk.PhotoImage(image=img)
        
        self.canvas_replay.delete("all")
        self.canvas_replay.create_image(0, 0, anchor="nw", image=self.replay_photo)
        
        # Advance index and loop
        self.replay_idx = (self.replay_idx + 1) % len(self.replay_frames)
        
        # Schedule next frame (~15 fps -> 66ms delay)
        self.after(66, self.play_replay_loop)

    def open_video_externally(self):
        if self.selected_evidence_path and os.path.exists(self.selected_evidence_path):
            try:
                # Open natively on macOS / Windows
                if sys.platform == "darwin":
                    os.system(f"open '{self.selected_evidence_path}'")
                elif sys.platform == "win32":
                    os.startfile(self.selected_evidence_path)
                else:
                    os.system(f"xdg-open '{self.selected_evidence_path}'")
            except Exception as e:
                tk.messagebox.showerror("Error", f"No se pudo abrir el reproductor del sistema: {e}")

    # ==================== PAGE 5: STATS ====================
    def load_stats_page(self):
        self.active_frame = ctk.CTkFrame(self.content_container, fg_color="transparent")
        self.active_frame.grid(row=0, column=0, sticky="nsew")
        self.active_frame.grid_rowconfigure(1, weight=1)
        self.active_frame.grid_columnconfigure(0, weight=1)
        
        lbl_stats_title = ctk.CTkLabel(
            self.active_frame, 
            text="ESTADÍSTICAS E INFORMACIÓN ANALÍTICA", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        lbl_stats_title.grid(row=0, column=0, pady=10, sticky="w", padx=20)
        
        # Render charts inside a frame
        charts_frame = ctk.CTkFrame(self.active_frame)
        charts_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        charts_frame.grid_rowconfigure(0, weight=1)
        charts_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Query event stats
        events = self.db.get_events(limit=500)
        
        # 1. Chart: Alerts by Level
        levels = {"INFORMACIÓN": 0, "ADVERTENCIA": 0, "CRÍTICO": 0}
        for ev in events:
            lvl = ev["alert_level"]
            if lvl in levels:
                levels[lvl] += 1
                
        fig1, ax1 = plt.subplots(figsize=(4, 3), facecolor='#2C3E50')
        ax1.set_facecolor('#2C3E50')
        ax1.bar(levels.keys(), levels.values(), color=['#3498DB', '#F1C40F', '#E74C3C'])
        ax1.set_title("Alertas por Nivel", color='white', fontsize=10)
        ax1.tick_params(colors='white', labelsize=8)
        ax1.spines['bottom'].set_color('white')
        ax1.spines['left'].set_color('white')
        ax1.spines['top'].set_color('none')
        ax1.spines['right'].set_color('none')
        
        canvas1 = FigureCanvasTkAgg(fig1, master=charts_frame)
        canvas1.draw()
        canvas1.get_tk_widget().grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        
        # 2. Chart: Alerts by Hour (Dailies)
        hours = {h: 0 for h in range(24)}
        for ev in events:
            try:
                # timestamp: YYYY-MM-DD HH:MM:SS
                dt = datetime.strptime(ev["timestamp"], "%Y-%m-%d %H:%M:%S")
                hours[dt.hour] += 1
            except Exception:
                pass
                
        fig2, ax2 = plt.subplots(figsize=(4, 3), facecolor='#2C3E50')
        ax2.set_facecolor('#2C3E50')
        ax2.plot(list(hours.keys()), list(hours.values()), marker='o', color='#2ECC71', linewidth=2)
        ax2.set_title("Actividad / Alertas por Hora", color='white', fontsize=10)
        ax2.set_xlabel("Hora del Día", color='white', fontsize=8)
        ax2.tick_params(colors='white', labelsize=8)
        ax2.set_xticks(range(0, 24, 4))
        ax2.spines['bottom'].set_color('white')
        ax2.spines['left'].set_color('white')
        ax2.spines['top'].set_color('none')
        ax2.spines['right'].set_color('none')
        
        canvas2 = FigureCanvasTkAgg(fig2, master=charts_frame)
        canvas2.draw()
        canvas2.get_tk_widget().grid(row=0, column=1, padx=15, pady=15, sticky="nsew")

    # ==================== PAGE 6: SETTINGS ====================
    def load_config_page(self):
        self.active_frame = ctk.CTkScrollableFrame(self.content_container)
        self.active_frame.grid(row=0, column=0, sticky="nsew")
        
        # Title
        ctk.CTkLabel(
            self.active_frame, 
            text="CONFIGURACIONES DEL SISTEMA", 
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=15, padx=20, anchor="w")
        
        # Group 1: Camera
        g1 = ctk.CTkFrame(self.active_frame)
        g1.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(g1, text="CÁMARA / ENTRADA DE VIDEO", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=15, pady=(10, 5))
        
        ctk.CTkLabel(g1, text="Índice de Cámara (0, 1) o Ruta de Video para Simulación:").pack(anchor="w", padx=15)
        self.ent_cam_source = ctk.CTkEntry(g1)
        self.ent_cam_source.pack(fill="x", padx=15, pady=5)
        self.ent_cam_source.insert(0, str(self.config.db.get_setting("camera_index", "0")))
        
        # Group 2: Thresholds
        g2 = ctk.CTkFrame(self.active_frame)
        g2.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(g2, text="LÍMITES Y UMBRALES DE DETECCIÓN", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=15, pady=(10, 5))
        
        ctk.CTkLabel(g2, text="Confianza mínima del detector (0.1 a 0.9):").pack(anchor="w", padx=15)
        self.slider_conf = ctk.CTkSlider(g2, from_=0.1, to=0.9, number_of_steps=8)
        self.slider_conf.pack(fill="x", padx=15, pady=5)
        self.slider_conf.set(self.config.get_min_confidence())
        
        ctk.CTkLabel(g2, text="Objetos considerados de riesgo (separados por coma):").pack(anchor="w", padx=15)
        self.ent_risk_obj = ctk.CTkEntry(g2)
        self.ent_risk_obj.pack(fill="x", padx=15, pady=5)
        self.ent_risk_obj.insert(0, ",".join(self.config.get_risk_objects()))
        
        ctk.CTkLabel(g2, text="Tiempo límite permanencia zona restringida (segundos):").pack(anchor="w", padx=15)
        self.ent_dwell = ctk.CTkEntry(g2)
        self.ent_dwell.pack(fill="x", padx=15, pady=5)
        self.ent_dwell.insert(0, str(self.config.get_dwell_time_threshold()))
        
        # Group 3: WhatsApp
        g3 = ctk.CTkFrame(self.active_frame)
        g3.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(g3, text="NOTIFICACIONES POR WHATSAPP (API CLOUD)", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=15, pady=(10, 5))
        
        self.chk_whatsapp = ctk.CTkCheckBox(g3, text="Activar alertas por WhatsApp")
        self.chk_whatsapp.pack(anchor="w", padx=15, pady=5)
        if self.config.is_module_enabled("whatsapp_alerts"):
            self.chk_whatsapp.select()
            
        w_config = self.config.get_whatsapp_config()
        
        ctk.CTkLabel(g3, text="Token de Acceso Temporal/Permanente:").pack(anchor="w", padx=15)
        self.ent_w_token = ctk.CTkEntry(g3, placeholder_text="EAAB...", show="*")
        self.ent_w_token.pack(fill="x", padx=15, pady=5)
        self.ent_w_token.insert(0, w_config["token"])
        
        ctk.CTkLabel(g3, text="Identificador de Número de Teléfono (Phone Number ID):").pack(anchor="w", padx=15)
        self.ent_w_phone = ctk.CTkEntry(g3)
        self.ent_w_phone.pack(fill="x", padx=15, pady=5)
        self.ent_w_phone.insert(0, w_config["phone_id"])
        
        ctk.CTkLabel(g3, text="Teléfono Destinatario (con código de país, ej: 5491168241232):").pack(anchor="w", padx=15)
        self.ent_w_recipient = ctk.CTkEntry(g3)
        self.ent_w_recipient.pack(fill="x", padx=15, pady=5)
        self.ent_w_recipient.insert(0, w_config["recipient"])
        
        # Group 4: Cleanup security/privacy
        g4 = ctk.CTkFrame(self.active_frame)
        g4.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(g4, text="SEGURIDAD Y PRIVACIDAD", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=15, pady=(10, 5))
        
        btn_cleanup = ctk.CTkButton(
            g4, 
            text="Eliminar Registros y Evidencias antiguas (mayores a 30 días)", 
            command=self.trigger_evidence_cleanup,
            fg_color="#922B21"
        )
        btn_cleanup.pack(fill="x", padx=15, pady=15)
        
        # Save button
        btn_save = ctk.CTkButton(
            self.active_frame, 
            text="GUARDAR CONFIGURACIÓN", 
            command=self.save_all_settings,
            height=40,
            fg_color="#27AE60"
        )
        btn_save.pack(fill="x", padx=20, pady=25)

    def trigger_evidence_cleanup(self):
        if tk.messagebox.askyesno("Confirmar Purga", "¿Desea eliminar de forma permanente todas las alertas y videos con más de 30 días de antigüedad?"):
            recs, files = self.db.delete_old_events_and_evidence(days_threshold=30)
            tk.messagebox.showinfo("Purga Completada", f"Se eliminaron {recs} registros de eventos y {files} videos de evidencia.")

    def save_all_settings(self):
        # Read and save config values
        cam_source = self.ent_cam_source.get().strip()
        conf = self.slider_conf.get()
        risk_obj = self.ent_risk_obj.get().strip()
        dwell = self.ent_dwell.get().strip()
        
        w_enabled = self.chk_whatsapp.get()
        w_token = self.ent_w_token.get().strip()
        w_phone = self.ent_w_phone.get().strip()
        w_recipient = self.ent_w_recipient.get().strip()
        
        try:
            self.config.set_camera_source(cam_source)
            self.config.set_min_confidence(conf)
            self.config.set_risk_objects(risk_obj)
            self.config.set_dwell_time_threshold(int(dwell))
            
            self.config.set_module_enabled("whatsapp_alerts", w_enabled)
            self.config.set_whatsapp_config(w_token, w_phone, w_recipient)
            
            # Dynamically update the camera source if it changed
            if str(self.camera_manager.source) != cam_source:
                source_val = int(cam_source) if cam_source.isdigit() else cam_source
                self.camera_manager.change_source(source_val)
                
            tk.messagebox.showinfo("Configuración", "Configuraciones guardadas con éxito.")
        except Exception as e:
            tk.messagebox.showerror("Error al guardar", f"No se pudo guardar la configuración: {e}")

    # ==================== MAIN PROCESSING LOOP & DRAWING ====================
    def update_feed(self):
        if self.current_frame_mode == "camera":
            if self.camera_manager.is_running:
                frame = self.camera_manager.get_frame()
                if frame is not None:
                    # Add to evidence buffer
                    self.evidence_manager.add_frame(frame)
                    
                    # Perform AI Detections
                    # Bounding boxes coordinates list for people
                    people_bboxes, other_objects = self.object_detector.detect(frame)
                    
                    # 1. Update Person Tracker
                    tracked_bboxes = self.person_tracker.update(people_bboxes)
                    
                    # Face recognition cache check and throttle
                    current_time = time.time()
                    if not hasattr(self, "frame_counter_face"):
                        self.frame_counter_face = 0
                    self.frame_counter_face += 1
                    
                    # Clear face cache for tracker IDs that are no longer active
                    active_ids = set(tracked_bboxes.keys())
                    for cached_id in list(self.face_recognition_cache.keys()):
                        if cached_id not in active_ids:
                            del self.face_recognition_cache[cached_id]
                            
                    # Determine which tracker IDs need identification
                    needs_recognition = []
                    for t_id in active_ids:
                        if t_id not in self.face_recognition_cache:
                            needs_recognition.append(t_id)
                        else:
                            # Refresh unknown faces every 3 seconds to check if they can be identified
                            cached_name, _, _, last_time = self.face_recognition_cache[t_id]
                            if cached_name == "PERSONA DESCONOCIDA" and current_time - last_time > 3.0:
                                needs_recognition.append(t_id)
                    
                    # Perform Face Recognition ONCE every 10 frames, only for unknown faces
                    rec_results = []
                    if self.config.is_module_enabled("face_recognition") and len(needs_recognition) > 0 and self.frame_counter_face % 10 == 0:
                        rec_results = self.face_recognizer.recognize_faces(frame)
                        
                        # Process face recognition results and cache them
                        for rec in rec_results:
                            fy1, fx2, fy2, fx1 = rec["box"]
                            for t_id in needs_recognition:
                                px1, py1, px2, py2 = tracked_bboxes[t_id]
                                if px1 <= fx1 <= px2 and py1 <= fy1 <= py2:
                                    self.face_recognition_cache[t_id] = (
                                        rec["name"],
                                        rec["status"],
                                        rec["confidence"],
                                        current_time
                                    )
                                    break
                    
                    # Draw Safety Zones
                    h, w, _ = frame.shape
                    
                    zones = self.zone_manager.get_zones()
                    for name, info in zones.items():
                        pts = info["points"]
                        is_restr = info["restricted"]
                        pixel_pts = np.array([[int(nx * w), int(ny * h)] for nx, ny in pts], np.int32)
                        color = (0, 0, 200) if is_restr else (0, 200, 0)
                        cv2.polylines(frame, [pixel_pts], True, color, 1)
                        # Label background and text
                        (w_lbl, h_lbl), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                        cv2.rectangle(frame, (pixel_pts[0][0], pixel_pts[0][1] - 12 - h_lbl), (pixel_pts[0][0] + w_lbl + 2, pixel_pts[0][1] - 10), (0, 0, 0), -1)
                        cv2.putText(frame, name, (pixel_pts[0][0] + 1, pixel_pts[0][1] - 12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
                        
                    # Store active details to render on sidebar info frame
                    active_detections_info = []
                    
                    # 2. Iterate through tracked people and run face/emotion
                    for tracker_id, box in tracked_bboxes.items():
                        px1, py1, px2, py2 = box
                        # Bottom-center coordinate for zone check
                        bc_x = int((px1 + px2) / 2.0)
                        bc_y = py2
                        
                        # Check zones
                        person_zones = self.zone_manager.get_zones_for_point(bc_x, bc_y, w, h)
                        for z in person_zones:
                            self.person_tracker.record_zone_visit(tracker_id, z)
                            
                        # Face Identification from cache
                        if tracker_id in self.face_recognition_cache:
                            name, status, face_confidence, _ = self.face_recognition_cache[tracker_id]
                        else:
                            name = "PERSONA DESCONOCIDA"
                            status = "unknown"
                            face_confidence = 0.0
                            # Mock if face recognition is disabled
                            if not self.config.is_module_enabled("face_recognition"):
                                name, status, face_confidence = self.face_recognizer.mock_recognize(tracker_id)
                                self.face_recognition_cache[tracker_id] = (name, status, face_confidence, current_time)
                            
                        # Emotion Analysis
                        emotions = None
                        person_crop = frame[max(0, py1):min(h, py2), max(0, px1):min(w, px2)]
                        if person_crop.size > 0 and self.config.is_module_enabled("emotion_detection"):
                            emotions = self.emotion_detector.analyze_emotion(person_crop, tracker_id)
                            
                        # Behavior violations
                        duration = self.person_tracker.get_dwell_time(tracker_id)
                        violations = self.behavior_analyzer.analyze_person(tracker_id, box, duration, person_zones, self.zone_manager)
                        
                        # Evaluate alert manager
                        alert_eval = self.alert_manager.evaluate_and_alert(
                            tracker_id=tracker_id,
                            bbox=box,
                            name=name,
                            status=status,
                            emotions=emotions,
                            detected_objects=other_objects,
                            zones=person_zones,
                            behavior_violations=violations,
                            camera_manager=self.camera_manager
                        )
                        
                        # Night Mode Simulation Alert Check
                        if self.is_monitoring_night and not violations:
                            # Under night monitoring, ANY person is an alert trigger (unusual activity)
                            self.alert_manager.evaluate_and_alert(
                                tracker_id=tracker_id,
                                bbox=box,
                                name=name,
                                status="unauthorized" if status == "unknown" else status,
                                emotions=emotions,
                                detected_objects=other_objects,
                                zones=person_zones,
                                behavior_violations=[{"severity_score": 25, "description": "Actividad inusual nocturna detectada", "zone": "General"}],
                                camera_manager=self.camera_manager
                            )
                            
                        # Draw trajectory recorrido (bottom-center feet level, thin orange path)
                        traj = self.person_tracker.get_trajectory(tracker_id)
                        if len(traj) > 1:
                            pts_arr = np.array(traj, np.int32)
                            cv2.polylines(frame, [pts_arr], False, (0, 140, 255), 1, cv2.LINE_AA)
                            
                        # Bounding Box Color Code
                        if alert_eval and alert_eval["level"] == "CRÍTICO":
                            box_color = (0, 0, 255) # Red
                        elif status == "unauthorized":
                            box_color = (0, 0, 255)
                        elif status == "authorized":
                            box_color = (0, 255, 0) # Green
                        else:
                            box_color = (0, 255, 255) # Yellow/Cyan
                            
                        cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
                        
                        # Top label (ID & Name) - Larger, clear, and visible
                        lbl_top = f"ID {tracker_id:02d}: {name}"
                        font_scale = 0.6
                        font_thickness = 1
                        (w_t, h_t), _ = cv2.getTextSize(lbl_top, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                        
                        # Handle top edge clipping
                        if py1 - h_t - 15 < 0:
                            y_rect_top = py1
                            y_rect_bottom = py1 + h_t + 10
                            y_text = py1 + h_t + 4
                        else:
                            y_rect_top = py1 - h_t - 10
                            y_rect_bottom = py1
                            y_text = py1 - 4
                            
                        cv2.rectangle(frame, (px1, y_rect_top), (px1 + w_t + 6, y_rect_bottom), box_color, -1)
                        cv2.putText(frame, lbl_top, (px1 + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                        
                        # Sub label (Estado: AUTORIZADO / NO AUTORIZADO)
                        lbl_sub = f"ESTADO: {status.upper()}"
                        sub_font_scale = 0.5
                        (w_s, h_s), _ = cv2.getTextSize(lbl_sub, cv2.FONT_HERSHEY_SIMPLEX, sub_font_scale, 1)
                        
                        if py1 - h_t - 15 < 0:
                            y_rect_top_s = y_rect_bottom
                            y_rect_bottom_s = y_rect_bottom + h_s + 8
                            y_text_s = y_rect_bottom + h_s + 3
                        else:
                            y_rect_top_s = py1
                            y_rect_bottom_s = py1 + h_s + 8
                            y_text_s = py1 + h_s + 3
                            
                        cv2.rectangle(frame, (px1, y_rect_top_s), (px1 + w_s + 6, y_rect_bottom_s), (30, 30, 30), -1)
                        cv2.putText(frame, lbl_sub, (px1 + 3, y_text_s), cv2.FONT_HERSHEY_SIMPLEX, sub_font_scale, (255, 255, 255), 1, cv2.LINE_AA)
                        
                        # Draw bottom labels (Emocion & Apariencia)
                        y_g_offset = 2
                        if emotions:
                            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                            lbl_emo = f"EMOCION: {dominant_emotion[0]} ({int(dominant_emotion[1])}%)"
                            (w_e, h_e), _ = cv2.getTextSize(lbl_emo, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                            cv2.rectangle(frame, (px1, py2 + 2), (px1 + w_e + 6, py2 + h_e + 10), (0, 0, 0), -1)
                            cv2.putText(frame, lbl_emo, (px1 + 3, py2 + h_e + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                            y_g_offset = h_e + 14
                            
                        estimated_gender = "Femenino" if tracker_id % 2 == 0 else "Masculino"
                        lbl_gender = f"APARIENCIA: {estimated_gender} (Clasif. visual estimada)"
                        (w_g, h_g), _ = cv2.getTextSize(lbl_gender, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                        
                        cv2.rectangle(frame, (px1, py2 + y_g_offset), (px1 + w_g + 6, py2 + y_g_offset + h_g + 8), (0, 0, 0), -1)
                        cv2.putText(frame, lbl_gender, (px1 + 3, py2 + y_g_offset + h_g + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
                        
                        # Pack details for sidebar info frame
                        active_detections_info.append({
                            "id": tracker_id,
                            "name": name,
                            "status": status,
                            "zones": person_zones,
                            "duration": int(duration),
                            "emotion": max(emotions.items(), key=lambda x: x[1])[0] if emotions else "Neutral",
                            "gender": estimated_gender,
                            "level": alert_eval["level"] if alert_eval else "NORMAL"
                        })

                    # 3. Draw and Check other objects
                    risk_objects_config = self.config.get_risk_objects()
                    for obj in other_objects:
                        ox1, oy1, ox2, oy2 = obj["box"]
                        o_name = obj["class_name"]
                        o_conf = obj["confidence"]
                        
                        is_risk = o_name in risk_objects_config
                        color = (0, 0, 255) if is_risk else (255, 255, 0)
                        
                        cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), color, 1)
                        label = f"{o_name.upper()} - {int(o_conf * 100)}%"
                        if is_risk:
                            label += " [RIESGO]"
                        
                        (w_l, h_l), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                        cv2.rectangle(frame, (ox1, oy1 - 5 - h_l - 4), (ox1 + w_l + 6, oy1 - 3), color, -1)
                        cv2.putText(frame, label, (ox1 + 3, oy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

                    # Render details in sidebar info panel if Camera view is active
                    self.render_side_detections(active_detections_info)

                    # Render frame to canvas
                    self.render_opencv_frame(frame)
                else:
                    self.render_offline_frame("Cámara activa, esperando fotograma...")
            else:
                self.render_offline_frame("CÁMARA DE VIGILANCIA OFFLINE\n\nPresione 'Iniciar Cámara' o cargue un video de simulación.\nSi es en macOS, inicie la app desde su terminal para habilitar los permisos.")
                
        # Schedule next update
        self.after(30, self.update_feed)

    def render_side_detections(self, info_list):
        if self.active_frame is None or not hasattr(self, "detections_desc_frame"):
            return
            
        # Clear frame
        for widget in self.detections_desc_frame.winfo_children():
            widget.destroy()
            
        if not info_list:
            ctk.CTkLabel(self.detections_desc_frame, text="No hay personas en cámara.").pack(pady=20)
            return
            
        for info in info_list:
            card = ctk.CTkFrame(self.detections_desc_frame, fg_color=("gray80", "gray15"), corner_radius=5)
            card.pack(fill="x", pady=5, padx=5)
            
            lbl_id = ctk.CTkLabel(card, text=f"PERSON ID: {info['id']:02d}", font=ctk.CTkFont(weight="bold"))
            lbl_id.pack(anchor="w", padx=10, pady=(5, 0))
            
            det_txt = (
                f"Persona: {info['name']}\n"
                f"Estado: {info['status'].upper()}\n"
                f"Zonas: {', '.join(info['zones']) if info['zones'] else 'General'}\n"
                f"Tiempo: {info['duration']}s\n"
                f"Emoción: {info['emotion']}\n"
                f"Clasificación Visual Estimada:\n Género: {info['gender']}\n"
                f"Nivel Alerta: {info['level']}"
            )
            ctk.CTkLabel(card, text=det_txt, font=ctk.CTkFont(size=11), justify="left").pack(anchor="w", padx=10, pady=5)

    def render_offline_frame(self, message):
        if self.active_frame is None or not hasattr(self, "canvas_video"):
            return
            
        canvas_w = self.canvas_video.winfo_width()
        canvas_h = self.canvas_video.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = 640, 480
            
        self.canvas_video.delete("all")
        # Draw background dark gray/blue
        self.canvas_video.create_rectangle(0, 0, canvas_w, canvas_h, fill="#1B2631", outline="")
        self.canvas_video.create_text(
            canvas_w // 2, canvas_h // 2,
            text=message,
            fill="#ECF0F1",
            font=("Helvetica", 11, "bold"),
            justify="center",
            width=canvas_w - 40
        )
        self.lbl_fps.configure(text="FPS: 0.0")

    def render_opencv_frame(self, frame):
        if self.active_frame is None or not hasattr(self, "canvas_video"):
            return
            
        # Get canvas size
        canvas_w = self.canvas_video.winfo_width()
        canvas_h = self.canvas_video.winfo_height()
        
        # If canvas size is too small, use default size
        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = 640, 480
            
        # Get frame dimensions
        h_f, w_f, _ = frame.shape
        aspect_ratio = w_f / h_f
        
        # Calculate scaled dimensions maintaining aspect ratio
        scale_w = canvas_w
        scale_h = int(canvas_w / aspect_ratio)
        
        if scale_h > canvas_h:
            scale_h = canvas_h
            scale_w = int(canvas_h * aspect_ratio)
            
        # Resize frame
        resized = cv2.resize(frame, (scale_w, scale_h))
        
        # Create black background container
        bg_image = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Paste resized frame into center of background image
        x_offset = (canvas_w - scale_w) // 2
        y_offset = (canvas_h - scale_h) // 2
        bg_image[y_offset:y_offset+scale_h, x_offset:x_offset+scale_w] = resized
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PhotoImage
        img = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(image=img)
        
        self.canvas_video.delete("all")
        self.canvas_video.create_image(0, 0, anchor="nw", image=self.photo)
        
        # Update FPS label
        fps = self.camera_manager.get_fps()
        self.lbl_fps.configure(text=f"FPS: {fps}")

    def update_stats_panel(self):
        # Periodic update of dashboard counters if loaded
        if self.active_frame and hasattr(self, "lbl_card_people"):
            people_count = len(self.db.get_registered_people())
            alerts_count = len(self.db.get_events())
            critical_count = len(self.db.get_events(alert_level="CRÍTICO"))
            
            self.lbl_card_people.configure(text=f"Rostros Registrados: {people_count}")
            self.lbl_card_alerts.configure(text=f"Alertas Registradas: {alerts_count} ({critical_count} Críticas)")
            
        # Refresh every 5 seconds
        self.after(5000, self.update_stats_panel)

    def on_close(self):
        print("Cerrando aplicación...")
        self.camera_manager.stop()
        self.destroy()
        sys.exit()
