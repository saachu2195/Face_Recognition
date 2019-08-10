#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os
os.chdir("C:/Users/square/Desktop/Face_Recognition_Detector")
import numpy as np


## Importing the Haar Cascade
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detector  = cv2.CascadeClassifier("haarcascade_eye.xml")


## Adding a video capture object
cap = cv2.VideoCapture(0)
id = input("Please Enter your User_ID: ")
sampleNum = 0
while(True):
    ## Capture frame by frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector.detectMultiScale(gray, scaleFactor = 1.4, minNeighbors = 5)
    print("Number of Faces", len(faces))
    
    for (x, y, w, h) in faces:         ## X & y are the location of rectangle.
        sampleNum = sampleNum+1
        roi_color = frame[y:y+h, x:x+w]
        cv2.imwrite("dataset/users." + str(id)+ "." + str(sampleNum) + ".jpg", roi_color)
        print(x, y, w, h)              ## w & h are the width & height.
        
         
        ## Drawing a Rectangle
        color = (200, 100, 50)
        stroke = 2
        width = x+w
        height = y+h
        cv2.rectangle(frame, (x, y), (width, height), color, stroke)
        
        ## For Eyes
        eyes = eye_detector.detectMultiScale(roi_color)
        for (ex, ey, ew, eh) in eyes:
            roi_eyecolor = frame[y:y+h, x:x+w]
            cv2.rectangle(roi_eyecolor, (ex,ey), (ex+ew, ey+eh), (0,255, 0), 2)
        cv2.imwrite("dataset/users." + str(id)+ "." + str(sampleNum) + ".jpg", roi_color)
        cv2.waitKey(100)
        
    ## Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    if(sampleNum > 15):
        break
        
## When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# In[ ]:




