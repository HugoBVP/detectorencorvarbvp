import os
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Inicialización de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para iniciar la cámara
@app.route('/camera')
def camera():
    return render_template('camera.html')

# Ruta para capturar video y procesar la cámara
def generate_video():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la imagen a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # Convertir la imagen de vuelta a BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Codificar la imagen en JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            # Enviar la imagen codificada a través de un flujo HTTP
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    cap.release()

# Ruta para mostrar el video en tiempo real
@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
