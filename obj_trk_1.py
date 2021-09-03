import cv2
import time
import math


video = cv2.VideoCapture("bb3.mp4")
#load tracker 
tracker = cv2.TrackerCSRT_create()

#read the first frame of the video
success,img = video.read()

#selct the bounding box on the image
bbox = cv2.selectROI("tracking",img,False)

#initialise the tracker on the img and the bounding box
tracker.init(img,bbox)


while True:
    check,img = video.read()   
    success,bbox = tracker.update(img)

    if success:
        drawBox(img,bbox)
    else:
        cv2.putText(img,"Lost",(75,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    cv2.imshow("result",img)
            
    key = cv2.waitKey(1)
    if key == ord('q'):
        print("Closing")
        break

video.release()
cv2.destroyALLwindows()