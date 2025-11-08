import face_recognition 
import cv2             
import numpy as np     
import os              
import time 

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

## 2 Cámara

video_capture = cv2.VideoCapture(0)
fotogramas_por_procesar = 4
contador_fotogramas = 0
loc_caras_actual = []
cod_caras_actual = []
nombres_caras_en_pantalla = []
tiempo_saludo = 0.0
tiempo_espera = 0.0
timepo_captura = 0.0

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

                if current_time > timepo_captura:
                    
                    temporizador_captura_fin = current_time + 5.0
                    (top, right, bottom, left) = loc_caras_actual[i]
                    
                    top_orig = int(top / escala)
                    right_orig = int(right / escala)
                    bottom_orig = int(bottom / escala)
                    left_orig = int(left / escala)
                    
                    cara_recortada = frame[top_orig:bottom_orig, left_orig:right_orig]
                    
                    timestamp = time.strftime("%Y%m%d_%H%M%S") # Ej: "20251107_113029"
                    nuevo_nombre_archivo = f"desconocido_{timestamp}.jpg"
                    
                    ruta_guardado = os.path.join(ruta_caras_conocidas, nuevo_nombre_archivo)

                    cv2.imwrite(ruta_guardado, cara_recortada)

                    print(f"¡Desconocido detectado! Foto guardada como: {nuevo_nombre_archivo}")

            nombres_caras_en_pantalla.append(nombre)

    contador_fotogramas += 1

## 4 Cuadro de texto

    for (top, right, bottom, left), nombre in zip(loc_caras_actual, nombres_caras_en_pantalla):

        top = int(top / escala)
        right = int(right / escala)
        bottom = int(bottom / escala)
        left = int(left / escala)
        color = (0, 255, 0) if nombre != "Desconocido" else (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(frame, nombre, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

    if current_time < tiempo_saludo: 

        cv2.putText(frame, "Holiwe", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 1)

    cv2.imshow('Reconocimiento Facial', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

## 5 Fin 

video_capture.release()
cv2.destroyAllWindows()

print("\nPrograma finalizado.")