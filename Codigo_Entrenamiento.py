import os
import shutil
import torch
from ultralytics import YOLO

def entrenar_modelo_pesado_rotado():
    if torch.cuda.is_available():
        dispositivo = 'cuda'
        print(f"Jala perfecto, usando la gráfica: {torch.cuda.get_device_name(0)}")
    else:
        dispositivo = 'cpu'
        print("Ojo: No está detectando tu gráfica.")

    # Cargamos el modelo Medium para mayor capacidad de análisis
    model = YOLO('yolo11m-seg.pt') 
    
    # Tu ruta exacta para los archivos
    ruta_base = r"C:\Users\jorge\Downloads\PROYECTODADOS"
    ruta_yaml = os.path.join(ruta_base, "data.yaml")
    
    print("Iniciando entrenamiento pesado con rotación aumentada. ¡A darle!")
    
    resultados = model.train(
        data=ruta_yaml,
        epochs=100,         
        imgsz=800,          
        batch=8, # Protege los 4GB de VRAM de la laptop          
        device=dispositivo, 
        workers=4,
        degrees=60.0, # Aumentamos el ángulo para generar más datos sintéticos en RAM     
        save=True           
    )

    # Rescatamos el archivo final de la carpeta oculta de YOLO
    ruta_modelo_yolo = os.path.join(resultados.save_dir, "weights", "best.pt")
    ruta_modelo_tuya = os.path.join(ruta_base, "modelo_pesado.pt")
    
    if os.path.exists(ruta_modelo_yolo):
        shutil.copy(ruta_modelo_yolo, ruta_modelo_tuya)
        print(f"¡Listo! El modelo definitivo se guardó directamente en: {ruta_modelo_tuya}")
    else:
        print("Ojo: No se encontró el archivo entrenado en las carpetas de origen.")

if __name__ == '__main__':
    entrenar_modelo_pesado_rotado()