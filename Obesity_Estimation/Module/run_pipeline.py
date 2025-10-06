import os
import subprocess

# --- Definición de la Secuencia de Comandos ---
# Lista de scripts a ejecutar en el orden correcto
pipeline_steps = [
    "notebooks/make_dataset.py",
    "models/train_model.py",
    "models/evaluate_model.py"
]

# --- Ejecución del Pipeline ---
print("🚀 Iniciando el pipeline de Machine Learning...")

for step in pipeline_steps:
    command = f"python {step}"
    print(f"\n▶️  EJECUTANDO PASO: {command}")
    
    # Usamos subprocess para tener más control y ver la salida en tiempo real
    # os.system(command) también es una opción más simple
    result = subprocess.run(command, shell=True)
    
    # Si un paso falla, detenemos todo el pipeline
    if result.returncode != 0:
        print(f"❌ ERROR: El script '{step}' falló. Abortando el pipeline.")
        break
else:
    # Este bloque 'else' solo se ejecuta si el bucle 'for' termina sin un 'break'
    print("\n✅ ¡Pipeline completado exitosamente!")