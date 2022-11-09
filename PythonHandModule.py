import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelCom = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelCom, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        frameColor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.handResults = self.hands.process(frameColor)

        if self.handResults.multi_hand_landmarks:
            for i in self.handResults.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, i, self.mpHands.HAND_CONNECTIONS)

        return frame

    def handPosition(self, frame, handNum=0, draw=True):

        lmList = []
        if self.handResults.multi_hand_landmarks:
            myHand = self.handResults.multi_hand_landmarks[handNum]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        return lmList


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    detector = handDetector()
    while True:
        val, frame = cap.read()
        frame = cv2.flip(frame,1)
        frame = detector.findHands(frame)
        lmList = detector.handPosition(frame)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, str(int(fps)), (10, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("test", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
