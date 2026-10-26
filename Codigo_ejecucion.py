import cv2
import time
import math
import numpy as np
from ultralytics import YOLO

model = YOLO(r"C:\Users\jorge\Downloads\PROYECTODADOS\modelo_pesado.pt") 

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Iniciando cámara")

for i in range(5, 0, -1):
    tiempo_inicio = time.time()
    while time.time() - tiempo_inicio < 1:
        ret, frame = cap.read()
        if not ret: break
        cv2.putText(frame, str(i), (550, 400), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), 20)
        cv2.imshow('Deteccion de Dados', frame)
        cv2.waitKey(1)

ret, frame_capturado = cap.read()
cap.release()
cv2.destroyAllWindows()

if ret:
    print("Foto guardada")
    
    resultados = model(frame_capturado, conf=0.4, imgsz=1024, augment=True)
    frame_pintado = resultados[0].plot(labels=False, conf=False) 
    
    suma_total = 0
    cajas = resultados[0].boxes

    if cajas is not None:
        for caja in cajas:
            clase_id = int(caja.cls[0])
            numero = int(model.names[clase_id])
            x1, y1, x2, y2 = caja.xyxy[0].cpu().numpy()
 
            if numero == 6 or numero == 9:
                margen = 15
                ex1, ey1 = max(0, int(x1) - margen), max(0, int(y1) - margen)
                ex2, ey2 = min(frame_capturado.shape[1], int(x2) + margen), min(frame_capturado.shape[0], int(y2) + margen)
                
                recorte = frame_capturado[ey1:ey2, ex1:ex2]
                gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
                
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                gris_clahe = clahe.apply(gris)
                
                _, umbral = cv2.threshold(gris_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contornos) >= 2:
                    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
                    c_num = contornos[0] 
                    
                    punto_valido = None

                    for c in contornos[1:]:
                        area = cv2.contourArea(c)
                        if 5 < area < 150: 
                            perimetro = cv2.arcLength(c, True)
                            if perimetro > 0:
                                circularidad = 4 * math.pi * (area / (perimetro * perimetro))
                                if circularidad > 0.4: 
                                    punto_valido = c
                                    break 
                    
                    if punto_valido is not None:
                        M_num = cv2.moments(c_num)
                        cx_num = int(M_num["m10"] / M_num["m00"]) if M_num["m00"] != 0 else 0
                        cy_num = int(M_num["m01"] / M_num["m00"]) if M_num["m00"] != 0 else 0
                        
                        M_pto = cv2.moments(punto_valido)
                        cx_pto = int(M_pto["m10"] / M_pto["m00"]) if M_pto["m00"] != 0 else 0
                        cy_pto = int(M_pto["m01"] / M_pto["m00"]) if M_pto["m00"] != 0 else 0
                        
                        distancia = math.hypot(cx_pto - cx_num, cy_pto - cy_num)
                        diagonal_caja = math.hypot(ex2 - ex1, ey2 - ey1)
                        proporcion = distancia / diagonal_caja
                        
                        if proporcion < 0.5: 
                            numero = 6
                        else:
                            numero = 9
            
            suma_total += numero
            texto = str(numero)
            
            cv2.putText(frame_pintado, texto, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
            cv2.putText(frame_pintado, texto, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(frame_pintado, f'Suma total: {suma_total}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 5)
    
    print("Presiona cualquier tecla en la imagen para cerrar")
    cv2.imshow('Resultado Final - Suma de Dados', frame_pintado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error no se pudo capturar la foto.")