import time

import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)  # open the inbuild camera or 1 for external camera
mpHands = mp.solutions.hands  # using handmodule to tracks the hands
hands = mpHands.Hands()  # for indexing the hand
mpDraw = mp.solutions.drawing_utils  # for drawing the connections of indexes

pTime = 0
cTime = 0

while True:
    val, frame = cap.read()  # It will take the camera as an input.
    handColor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # It will color the indexes.
    handResults = hands.process(handColor)  # It will precess and show in the frame.
    # multi_hand_landmarks helps to stable all the index on hand.
    if handResults.multi_hand_landmarks:
        for i in handResults.multi_hand_landmarks:
            for id, lm in enumerate(i.landmark):
                h, w, c = frame.shape  # (h=Height, w=Width, c=Center) on the frame
                cx, cy = int(lm.x * w), int(lm.y * h)  # It will calculate and find the center of the screen.
                if id == 4:
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 255),
                               cv2.FILLED)  # Creating circle at the tip of the thumbfinger.
                mpDraw.draw_landmarks(frame, i, mpHands.HAND_CONNECTIONS)  # shows the connection of each indexes.
    # for calculating the fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (10, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("test", frame)  # Name of the frame
    cv2.waitKey(1)
