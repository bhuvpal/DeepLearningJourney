import time
import mediapipe as mp
import numpy as np

import handTrackingModule as Htm
import cv2
import math
import pyCoW
from pycaw.pycaw import AudioUtilities
import numpy



device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
print(f"Audio output: {device.FriendlyName}")
print(f"- Muted: {bool(volume.GetMute())}")
print(f"- Volume level: {volume.GetMasterVolumeLevel()} dB")
print(f"- Volume range: {volume.GetVolumeRange()[0]} dB - {volume.GetVolumeRange()[1]} dB")
#volume.SetMasterVolumeLevel(0, None)
vol = volume.GetVolumeRange()
minVol = vol[0]
maxVol = vol[1]
#print(volume.SetMasterVolumeLevel(maxVol,None))

cap = cv2.VideoCapture('2.mp4')
wCam, hCam = (640,480)

cap.set(3,wCam)
cap.set(4,hCam)
pTime = 0
detector = Htm.handDetector(detectionCon=0.7)
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    if len(lmList)!=0:
    #     print(lmList[0][4][1],lmList[0][4][2])
        #print(lmList[4],lmList[8])
        x1, y1 = lmList[0][4][1], lmList[0][4][2]
        x2, y2 = lmList[0][8][1], lmList[0][8][2]
        cv2.circle(img,(x1, y1),10,(255,255,0),cv2.FILLED)
        cv2.circle(img,(x2, y2),10,(255,255,0),cv2.FILLED)
        cv2.line(img,(x1, y1),(x2, y2),(255,255,0),3,cv2.FILLED)
        cx, cy = int((x1 + x2)//2), int((y1 + y2)//2)
        cv2.circle(img,(cx, cy),10,(255,255,0),cv2.FILLED)
        length = math.hypot((x2 - x1),(y2 - y1))
        #print(length)
        # volume range (50-300)
        # convert Volume range (-65.0,0)
        vol = np.interp(length,[50,200],[minVol,maxVol])
        volume.SetMasterVolumeLevel(vol, None)
        print(vol)
        if length < 50:
            cv2.circle(img,(cx, cy),12,(0,255,0),cv2.FILLED)




    #cv2.line(img,(x1, y1),(x2, y2),(0,0,255),3)




    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(50,70),3,cv2.FONT_HERSHEY_PLAIN,(235, 206, 135),1)
    cv2.imshow("Image",img)
    cv2.waitKey(10)