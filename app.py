from flask import Flask, Response, render_template, request
import cv2
import mediapipe as mp
import numpy as np
import time

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Variables globales
calibrated_angles = {}
calibration_complete = False
calibration_time = 0  # Para mostrar el mensaje de calibración exitosa

# Función para calcular el ángulo entre tres puntos
def calculate_angle(a, b, c):
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Calibrar ángulos naturales de la DIP
def calibrate(hand_landmarks):
    global calibrated_angles
    fingers = [8, 12, 16, 20]  # Índices de los dedos
    for finger in fingers:
        mcp = hand_landmarks[finger - 3]
        pip = hand_landmarks[finger - 2]
        dip = hand_landmarks[finger - 1]
        calibrated_angles[finger] = calculate_angle(mcp, pip, dip)
    print("Calibración completada. Ángulos calibrados:", calibrated_angles)

# Analizar encorvamiento y dibujar en el fotograma
def analyze_and_draw(frame, hand_landmarks):
    global calibrated_angles
    h, w, _ = frame.shape
    fingers = [8, 12, 16, 20]
    for finger in fingers:
        mcp = hand_landmarks[finger - 3]
        pip = hand_landmarks[finger - 2]
        dip = hand_landmarks[finger - 1]
        tip = hand_landmarks[finger]

        current_angle = calculate_angle(mcp, pip, dip)

        # Detectar encorvamiento si el ángulo actual es menor al calibrado (96% de sensibilidad)
        if current_angle < 0.8 * calibrated_angles.get(finger, 180):
            # Dibujar el dedo en rojo
            cv2.line(frame, (int(tip.x * w), int(tip.y * h)), (int(dip.x * w), int(dip.y * h)), (0, 0, 255), 3)
            cv2.line(frame, (int(dip.x * w), int(dip.y * h)), (int(pip.x * w), int(pip.y * h)), (0, 0, 255), 3)

            # Agregar mensaje en la pantalla
            message = f"Dedo {finger // 4}: Encorvado!"
            cv2.putText(frame, message, (10, 50 + 30 * (finger // 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Crear aplicación Flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    global calibration_complete, calibration_time
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                if not calibration_complete:
                    cv2.putText(frame, "Presiona 'Calibrar' para iniciar", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    analyze_and_draw(frame, hand_landmarks.landmark)
                    if calibration_complete and (time.time() - calibration_time < 2):
                        cv2.putText(frame, "Calibración exitosa", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/calibrate', methods=['POST'])
def handle_calibration():
    global calibration_complete, calibration_time
    calibration_complete = True
    calibration_time = time.time()
    return "Calibración activada"

if __name__ == '__main__':
    app.run(debug=True)
