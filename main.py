import os
import sys
from config import AppConfig
from dashboard.dashboard import SecurityDashboard

def main():
    print("==================================================")
    print("               AI SECURITY VISION                 ")
    print("==================================================")
    print("Iniciando sistema de videovigilancia inteligente...")
    
    # Ensure logs and data directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/known_faces", exist_ok=True)
    os.makedirs("data/alerts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Initialize configuration and database
    try:
        config = AppConfig()
        print("Base de datos y configuraciones inicializadas.")
    except Exception as e:
        print(f"Error crítico al inicializar la base de datos: {e}")
        sys.exit(1)

    # Launch GUI Dashboard
    try:
        app = SecurityDashboard(config)
        app.mainloop()
    except Exception as e:
        print(f"Error crítico en la interfaz gráfica: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
