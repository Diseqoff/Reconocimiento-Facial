import face_recognition 
import cv2             
import numpy as np     
import os              
import time 
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

## 1 Carga de la huellas de las caras

nombres_caras_conocidas = []
codificaciones_caras_conocidas = []
ruta_caras_conocidas = r"C:\Users\Nick\Desktop\Visual Studio Code\Programas\Python\Programas\Reconocimiento\caras_conocidas"

for nombre_archivo in os.listdir(ruta_caras_conocidas):

    if nombre_archivo.endswith((".jpg", ".png", ".jpeg")):
    
        ruta_imagen = os.path.join(ruta_caras_conocidas, nombre_archivo)
        imagen = face_recognition.load_image_file(ruta_imagen)
        codificaciones = face_recognition.face_encodings(imagen)

        if codificaciones:

            codificaciones_caras_conocidas.append(codificaciones[0])
            nombre = os.path.splitext(nombre_archivo)[0]
            nombres_caras_conocidas.append(nombre)

            print(f"Enrolada cara de: {nombre}")

        else:

            print(f"No se detectó ninguna cara en: {nombre_archivo}. Asegúrate de que la cara esté visible.")

print(f"\nTotal de caras conocidas cargadas: {len(nombres_caras_conocidas)}\n")

## 1.2 Carga de DNI 

db_dnis = {}
ruta_dnis_conocidos = r"C:\Users\Nick\Desktop\Visual Studio Code\Programas\Python\Programas\Reconocimiento\dnis_conocidos"

if not os.path.exists(ruta_dnis_conocidos):
    os.makedirs(ruta_dnis_conocidos)

for nombre_archivo in os.listdir(ruta_dnis_conocidos):

    if "_" in nombre_archivo:
            
        nombre_sin_ext = os.path.splitext(nombre_archivo)[0]
        partes = nombre_sin_ext.split("_")

        if len(partes) >= 2: 

            numero_dni = partes[0]
            nombre_persona = partes[1]
            db_dnis[numero_dni] = nombre_persona

            print(f"DNI Registrado: {numero_dni} - Nombre: {nombre_persona}")

print(f"\nTotal de DNIs conocidos cargados: {len(db_dnis)}\n")

## 2 Cámara

video_capture = cv2.VideoCapture(0)
fotogramas_por_procesar = 4
contador_fotogramas = 0
loc_caras_actual = []
cod_caras_actual = []
nombres_caras_en_pantalla = []
huellas_desconocidos_temporal = []
fotos_tomadas_desconocidos = []
tiempo_saludo = 0.0
tiempo_espera = 0.0
timepo_captura = 0.0
puntos = False
modo_dni = False
modo_distancia = False
dni_detectado = "Esperando ..."
nombre_dni_actual = ""
nombre_para_saludo = ""

## 3 Bucle que compara las huellas faciales

while True:

    ret, frame = video_capture.read()
    current_time = time.time()

    if not ret:

        print("Error: No se pudo capturar el fotograma de la cámara.")
        break

    if contador_fotogramas % fotogramas_por_procesar == 0:

        escala = 0.25
        fotograma_pequeno = cv2.resize(frame, (0, 0), fx=escala, fy=escala)
        rgb_fotograma_pequeno = cv2.cvtColor(fotograma_pequeno, cv2.COLOR_BGR2RGB)
        loc_caras_actual = face_recognition.face_locations(rgb_fotograma_pequeno)
        cod_caras_actual = face_recognition.face_encodings(rgb_fotograma_pequeno, loc_caras_actual)
        nombres_caras_en_pantalla = []
        marks_list = []

        if puntos:
            
            marks_list = face_recognition.face_landmarks(rgb_fotograma_pequeno, loc_caras_actual)

        for i, codificacion_cara_actual in enumerate(cod_caras_actual):

            coincidencias = face_recognition.compare_faces(codificaciones_caras_conocidas, codificacion_cara_actual, tolerance=0.6)
            nombre = "Desconocido" 

            if True in coincidencias:

                primer_indice_coincidencia = coincidencias.index(True)
                nombre = nombres_caras_conocidas[primer_indice_coincidencia]

                if current_time > tiempo_espera:
                    
                    print(f"Texto activado: Holiwe")

                    tiempo_saludo = current_time + 5.0
                    tiempo_espera = current_time + 10.0

            else:

                coincidencia_desconocido = face_recognition.compare_faces(huellas_desconocidos_temporal, codificacion_cara_actual, tolerance=0.5)
                indice_desconocido = -1

                if True in coincidencia_desconocido:

                    indice_desconocido = coincidencia_desconocido.index(True)

                if current_time > timepo_captura:
                    
                    tomar_foto = False

                    if indice_desconocido == -1:

                        print("Desconocido nuevo detectado")

                        huellas_desconocidos_temporal.append(codificacion_cara_actual)
                        fotos_tomadas_desconocidos.append(1)
                        tomar_foto = True
                        indice_desconocido = len(huellas_desconocidos_temporal) - 1

                    else:

                        fotos_actuales = fotos_tomadas_desconocidos[indice_desconocido]

                        if fotos_actuales < 3:

                            print(f"Visitante recurrente. Foto {fotos_actuales + 1} de 3.")

                            fotos_tomadas_desconocidos[indice_desconocido] += 1
                            tomar_foto = True
                    
                    if tomar_foto:

                        temporizador_captura_fin = current_time + 3.0
                        (top, right, bottom, left) = loc_caras_actual[i]
                    
                        top_orig = int(top / escala)
                        right_orig = int(right / escala)
                        bottom_orig = int(bottom / escala)
                        left_orig = int(left / escala)

                        top_orig = max(0, top_orig)
                        left_orig = max(0, left_orig)
                        bottom_orig = min(frame.shape[0], bottom_orig)
                        right_orig = min(frame.shape[1], right_orig)
                    
                        cara_recortada = frame[top_orig:bottom_orig, left_orig:right_orig]

                        if cara_recortada.size > 0:                    
                    
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            id_real = indice_desconocido if indice_desconocido != -1 else len(huellas_desconocidos_temporal) - 1
                            nuevo_nombre_archivo = f"desconocido_ID{id_real}_{timestamp}.jpg"
                            ruta_guardado = os.path.join(ruta_caras_conocidas, nuevo_nombre_archivo)

                            cv2.imwrite(ruta_guardado, cara_recortada)

                            print(f"Foto guardada como: {nuevo_nombre_archivo}")

            nombres_caras_en_pantalla.append(nombre)

    contador_fotogramas += 1

## 4 Cuadro de texto

    for i, (coords, nombre) in enumerate(zip(loc_caras_actual, nombres_caras_en_pantalla)):

        (top, right, bottom, left) = coords

        top = int(top / escala)
        right = int(right / escala)
        bottom = int(bottom / escala)
        left = int(left / escala)
        color = (0, 255, 0) if nombre != "Desconocido" else (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)

## 4.1 Modo distancia

        texto_mostrar = nombre

        if modo_distancia:

            alto_cara_pixeles = bottom - top
            alto_real_cm = 20
            focal_camara = 370

            if alto_cara_pixeles > 0:

                distancia_metro = (alto_real_cm * focal_camara) / alto_cara_pixeles / 100
                texto_mostrar = f"{nombre} ({distancia_metro:.2f}m)"

        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(frame, texto_mostrar, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

## 4.2 Puntos faciales

        if puntos and i < len(marks_list):

            puntos_actuales = marks_list[i]

            for rasgo_facial in puntos_actuales.values():
                
                for j in range(len(rasgo_facial)):

                    point_1 = rasgo_facial[j]

                    x1 = int(point_1[0] / escala)
                    y1 = int(point_1[1] / escala)

                    cv2.circle(frame, (x1, y1), 1, (0, 255, 0), -1)

                    if j < len(rasgo_facial) - 1:

                        point_2 = rasgo_facial[j + 1]

                        x2 = int(point_2[0] / escala)
                        y2 = int(point_2[1] / escala)

                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

## 4.3 DNI

    if modo_dni:

        h, w, _ = frame.shape
        cv2.rectangle(frame, (w//2 - 150, h//2 - 100), (w//2 + 150, h//2 + 100), (255, 255, 0), 2)
        cv2.putText(frame, "Coloca DNI aqui", (w//2 - 140, h//2 - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if contador_fotogramas % 15 == 0:

            try:

                zona_dni = frame[h//2 - 100:h//2 + 100, w//2 - 150:w//2 + 150]
                griss = cv2.cvtColor(zona_dni, cv2.COLOR_BGR2GRAY)
                _, umbral = cv2.threshold(griss, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                texto = pytesseract.image_to_string(umbral)
                busqueda_dni = re.search(r'\b\d{8}\b', texto)

                if busqueda_dni:

                    dni_detectado = busqueda_dni.group(0)
                    print(f"DNI detectado: {dni_detectado}")

                    if dni_detectado in db_dnis:

                        dni_detectado = f"{dni_detectado} - {db_dnis[dni_detectado]}"

                        if current_time > tiempo_espera:

                            nombre_para_saludo = db_dnis[busqueda_dni.group(0)]
                            tiempo_saludo = current_time + 5.0
                            tiempo_espera = current_time + 10.0

                    else:

                        dni_detectado = f"{dni_detectado} - No registrado"

            except Exception as e:

                print(f"Error al procesar el DNI: {e}")

        cv2.putText(frame, f"DNI: {dni_detectado}", (50, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3) 

## 4.4 Saludo  

    if current_time < tiempo_saludo: 

        mensaje= f"Hola {nombre_para_saludo}"
        cv2.putText(frame, mensaje, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 0, 0), 1)

    cv2.imshow('Reconocimiento Facial', frame)

## 5 Teclas

    key = cv2.waitKey(1) & 0xFF

    if key == ord('f'):
        
        break

    if key == ord('p'):

        print("Modo puntos faciales:", "Activado" if not puntos else "Desactivado")
        
        puntos = not puntos
    
    if key == ord('d'):

        modo_dni = not modo_dni
        dni_detectado = "Esperando ..."
        print(f"Modo DNI:{'ON' if modo_dni else 'OFF'}")

    if key == ord('m'):

        modo_distancia = not modo_distancia
        print(f"Modo Distancia:{'ON' if modo_distancia else 'OFF'}")

video_capture.release()
cv2.destroyAllWindows()

print("\nPrograma finalizado.")