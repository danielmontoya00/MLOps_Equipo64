import os
import subprocess
from datetime import datetime

# --- CONFIGURACIÓN DE BASE ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # /Obesity_Estimation/
os.chdir(BASE_DIR)  # Cambiar directorio de trabajo al BASE_DIR

# --- Definición de la Secuencia de Comandos ---
# Lista de scripts a ejecutar en el orden correcto
pipeline_steps = [
    "notebooks/make_dataset.py",
    "models/train_model.py",
    "models/evaluate_model.py"
]

# --- CREAR CARPETA DE LOGS ---
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")


# --- Ejecución del Pipeline ---
print("🚀 Iniciando el pipeline de Machine Learning...")
print(f" Guardando registro en: {log_file}\n")

with open(log_file, "w") as log:
    log.write("=== PIPELINE DE MACHINE LEARNING ===\n")
    log.write(f"Inicio: {datetime.now()}\n\n")

    for step in pipeline_steps:
        command = f"python {step}"
        print(f"\n▶️  Ejecutando: {command}")
        log.write(f"\n--- Ejecutando paso: {command} ---\n")

        # Ejecuta el comando y guarda el resultado
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Escribe en el log tanto la salida como errores
        log.write(result.stdout)
        if result.stderr:
            log.write("\n[ERROR]:\n")
            log.write(result.stderr)

        # Si falla, detener el pipeline
        if result.returncode != 0:
            print(f"ERROR: el script '{step}' falló. Revisa el log en {log_file}")
            log.write(f"\nERROR: el script '{step}' falló.\n")
            break
    else:
        # Se ejecuta solo si no hubo 'break' en el ciclo
        print("\n¡Pipeline completado exitosamente!")
        log.write("\n¡Pipeline completado exitosamente!\n")

    log.write(f"\nFin: {datetime.now()}\n")