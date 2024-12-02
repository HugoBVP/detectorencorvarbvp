from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import time

app = Flask(__name__)

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Calibración inicial
calibrated_angles = {}
calibration_complete = False
calibration_time = 0  # Para controlar el tiempo de visualización del mensaje de calibración

# Función para calcular el ángulo entre tres puntos
def calculate_angle(a, b, c):
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Asegura que el valor esté dentro de [-1, 1]
    return np.degrees(angle)

# Calibrar ángulos naturales de la DIP (sin considerar el tip)
def calibrate(hand_landmarks):
    global calibrated_angles
    fingers = [8, 12, 16, 20]  # Puntas de los dedos (sin usar el tip)
    for finger in fingers:
        mcp = hand_landmarks[finger - 3]  # MCP
        pip = hand_landmarks[finger - 2]  # PIP
        dip = hand_landmarks[finger - 1]  # DIP
        calibrated_angles[finger] = calculate_angle(mcp, pip, dip)
    print("Calibración completada: Ángulos naturales guardados.")

# Detectar encorvamiento (sin considerar el tip)
def analyze_and_draw(image, hand_landmarks):
    global calibrated_angles
    h, w, _ = image.shape

    fingers = [8, 12, 16, 20]  # Puntas de los dedos
    for finger in fingers:
        mcp = hand_landmarks[finger - 3]  # MCP
        pip = hand_landmarks[finger - 2]  # PIP
        dip = hand_landmarks[finger - 1]  # DIP
        tip = hand_landmarks[finger]  # Tip (punta del dedo)

        # Calcular el ángulo actual
        current_angle = calculate_angle(mcp, pip, dip)

        # Detectar encorvamiento si el ángulo actual es significativamente menor al calibrado (96% de sensibilidad)
        if current_angle < 0.96 * calibrated_angles[finger]:  # Ajuste de sensibilidad a 96%
            # Dibujar el dedo en rojo desde tip hasta DIP
            tip_coords = (int(tip.x * w), int(tip.y * h))
            dip_coords = (int(dip.x * w), int(dip.y * h))
            cv2.line(image, tip_coords, dip_coords, (0, 0, 255), 3)

            # Dibujar el dedo en rojo desde DIP hasta PIP
            pip_coords = (int(pip.x * w), int(pip.y * h))
            cv2.line(image, dip_coords, pip_coords, (0, 0, 255), 3)

            # Crear una caja roja con bordes redondeados para el mensaje de error
            message = f"Dedo {finger // 4}: Encorvado!"
            box_color = (0, 0, 255)  # Rojo
            text_color = (255, 255, 255)  # Blanco

            # Tamaño del texto
            font_scale = 0.8
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            (w_text, h_text), _ = cv2.getTextSize(message, font, font_scale, thickness)

            # Coordenadas de la caja alrededor del texto
            padding = 10
            box_coords = ((10, 30 + 30 * (finger // 4)), (10 + w_text + 2 * padding, 30 + 30 * (finger // 4) + h_text + 2 * padding))
            top_left, bottom_right = box_coords

            # Dibujar la caja con bordes redondeados utilizando elipse en las esquinas
            cv2.rectangle(image, top_left, bottom_right, box_color, -1)  # Fondo de la caja
            radius = 15  # Radio para las esquinas redondeadas
            cv2.ellipse(image, (top_left[0] + radius, top_left[1] + radius), (radius, radius), 180, 0, 90, box_color, -1)
            cv2.ellipse(image, (top_left[0] + radius, bottom_right[1] - radius), (radius, radius), 270, 0, 90, box_color, -1)
            cv2.ellipse(image, (bottom_right[0] - radius, top_left[1] + radius), (radius, radius), 90, 0, 90, box_color, -1)
            cv2.ellipse(image, (bottom_right[0] - radius, bottom_right[1] - radius), (radius, radius), 0, 0, 90, box_color, -1)

            # Dibujar el texto dentro de la caja
            cv2.putText(image, message, (top_left[0] + padding, top_left[1] + padding + h_text), font, font_scale, text_color, thickness)

# Captura de cámara
cap = cv2.VideoCapture(0)

@app.route('/')
def home():
    return render_template('index.html')  # Página de inicio

@app.route('/start-camera')
def start_camera():
    global calibration_complete, calibration_time
    calibration_complete = False  # Reiniciar la calibración cuando se inicia la cámara
    calibration_time = 0
    return render_template('camera.html')  # O la página donde se muestra la cámara

# Generar un flujo de video en tiempo real
def gen():
    global cap, hands, calibration_complete, calibration_time
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Invertir la imagen para una vista lateral
        frame = cv2.flip(frame, 1)

        # Convertir a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar la imagen con MediaPipe Hands
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                if not calibration_complete:
                    cv2.putText(frame, "Coloca la mano en posición natural y toca la pantalla o presiona 'c' para calibrar", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    cv2.putText(frame, "Calibracion exitosa. Analizando...", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    analyze_and_draw(frame, hand_landmarks.landmark)

                    # Mostrar mensaje de calibración exitosa durante 2 segundos
                    if calibration_complete and (time.time() - calibration_time < 2):
                        height, width, _ = frame.shape
                        cv2.putText(frame, "Calibracion exitosa", (width // 4, height // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

        # Codificar la imagen en JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            break
        # Convertir la imagen a bytes y devolverla como una respuesta HTTP
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
