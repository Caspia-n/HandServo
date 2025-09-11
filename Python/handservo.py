import cv2
import mediapipe as mp
import serial
import time

# === Setup ===
port = "COM5"
baud = 115200
esp = serial.Serial(port, baud, timeout=1)
time.sleep(2)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(2)

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def distance(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2) ** 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Finger tip indices: index=8, middle=12, ring=16, thumb=4
            thumbtip = hand_landmarks.landmark[4]
            indexdip = hand_landmarks.landmark[7]
            indextip = hand_landmarks.landmark[8]
            indexmcp = hand_landmarks.landmark[5]
            middletip = hand_landmarks.landmark[12]
            middlemcp = hand_landmarks.landmark[9]
            ringtip = hand_landmarks.landmark[16]
            ringmcp = hand_landmarks.landmark[13]
            pinkytip = hand_landmarks.landmark[20]
            pinkymcp = hand_landmarks.landmark[17]

            # Distances
            d_index = distance(indextip, indexmcp)
            d_middle = distance(middletip, middlemcp)
            d_ring = distance(ringtip, ringmcp)
            d_pinky = distance(pinkytip, pinkymcp)
            d_thumb = distance(indexdip, thumbtip)

            # Map distances → servo angle
            def map_angle(d):
                angle_index = int(min(max((d - 0.02) * 1000, 0), 180))
                return angle_index
            a1 = map_angle(d_index)   # Servo 1
            a2 = map_angle(d_middle)  # Servo 2
            a3 = map_angle(d_ring)    # Servo 3
            a4 = map_angle(d_pinky)   # Servo 4
            a5 = map_angle(d_thumb)   # Servo 5

            # Send to ESP32
            msg = f"{a1},{a2},{a3},{a4},{a5}\n"
            esp.write(msg.encode())

            # Show values on screen
            cv2.putText(frame, msg.strip(), (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
esp.close()
cv2.destroyAllWindows()
