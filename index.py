import cv2
import numpy as np
import mediapipe as mp

# Initialize
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

canvas = np.zeros((480, 640, 3), dtype=np.uint8)  # Drawing canvas

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Mirror the image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            # Index finger tip is landmark 8
            x, y = lmList[8]
            cv2.circle(frame, (x, y), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(canvas, (x, y), 5, (255, 0, 0), -1)  # Draw on canvas

    # Combine canvas with webcam feed
    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, imgInv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("Air Canvas", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
