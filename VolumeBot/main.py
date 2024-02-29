import cv2
import mediapipe as mp
from math import hypot
import subprocess
import numpy as np

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

volbar = 400
volper = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

    if lmList != []:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cv2.circle(img, (x1, y1), 13, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 13, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        length = hypot(x2 - x1, y2 - y1)
        vol = int(np.interp(length, [5, 150], [0, 100]))  # Adjusted range for smaller hands

        # Set the volume using amixer command-line tool
        subprocess.run(["amixer", "-D", "pulse", "sset", "Master", f"{vol}%"])

        volbar = int(np.interp(length, [5, 150], [400, 150]))  # Adjusted range for smaller hands
        volper = int(np.interp(length, [5, 150], [0, 100]))    # Adjusted range for smaller hands

        print(vol, int(length))

    cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 4)
    cv2.rectangle(img, (50, volbar), (85, 400), (0, 0, 255), cv2.FILLED)
    cv2.putText(img, f"{volper}%", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 98), 3)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xff == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
