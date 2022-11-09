import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
handGesture = mp.solutions.hands.Hands()
drawUtils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_y = 0
index_x = 0
frame_width= 1280
frame_height = 720

while True:
    _,frame = cap.read()
    frame = cv2.flip(frame,1)
    frame_width, frame_height,_ = frame.shape
    rgbFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    output = handGesture.process(rgbFrame)
    hands = output.multi_hand_landmarks
    if hands: 
        for i in hands:
            drawUtils.draw_landmarks(frame, i)
            landMarks = i.landmark
            for id, landmarks in enumerate(landMarks):
                x = int(landmarks.x*frame_width)
                y = int(landmarks.y*frame_height)
                if id == 8:
                    cv2.circle(img=frame,center=(x,y),radius=20,color=(0,255,255))
                    index_x = screen_width/frame_width*x
                    index_y = screen_height/frame_height*y
                    pyautogui.moveTo(index_x,index_y)
                if id == 4:
                    cv2.circle(img=frame,center=(x,y),radius=20,color=(0,255,0))
                    thumb_x = screen_width/frame_width*x
                    thumb_y = screen_height/frame_height*y
                if id == 11:
                    cv2.circle(img=frame,center=(x,y),radius=20,color=(0,255,255))
                    middle_x = screen_width/frame_width*x
                    middle_y = screen_height/frame_height*y
                    print("x: ", abs(index_x - middle_x))
                    print("y: ", abs(index_y - middle_y))
                    if (abs(index_y - middle_y)<=30) or (abs(index_x - middle_x)<=50):
                        pyautogui.click()
                        pyautogui.sleep(1)
    cv2.imshow('Virtual Mouse',frame)
    cv2.waitKey(1)