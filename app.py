import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, request
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Configuración para mediaPipe y la detección de postura
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para la cámara (con MediaPipe y detección de encorvamiento)
@app.route('/camera')
def camera():
    return render_template('camera.html')

# Función para procesar la imagen y hacer la detección
def process_frame(frame):
    # Convertir la imagen de BGR a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Hacer la detección de postura
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Dibujar los puntos de la postura
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Aquí puedes agregar lógica para evaluar encorvamiento basado en los puntos clave de la postura

    return frame

# Ruta para recibir la imagen y procesarla
@app.route('/process', methods=['POST'])
def process_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join("uploads", filename)
            file.save(file_path)
            
            # Cargar la imagen desde el archivo
            image = cv2.imread(file_path)
            
            # Procesar la imagen
            processed_image = process_frame(image)
            
            # Convertir la imagen procesada a un formato que pueda ser enviado al cliente
            _, img_encoded = cv2.imencode('.jpg', processed_image)
            img_bytes = img_encoded.tobytes()
            
            return Response(img_bytes, mimetype='image/jpeg')

# Función para generar la transmisión en vivo desde la cámara
def gen_frames():
    # Abrir la cámara (0 es el índice predeterminado)
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Procesar la imagen con MediaPipe
        frame = process_frame(frame)

        # Convertir la imagen a JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame_bytes = buffer.tobytes()

        # Generar una imagen JPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# Ruta para la transmisión en vivo de la cámara
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
