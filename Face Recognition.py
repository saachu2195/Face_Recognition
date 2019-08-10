#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Importing the require basic libraries.

import cv2                  ## for building the face recognition.
import pandas as pd
import numpy as np
import os               ## for changing the directory.


# In[2]:


## For Face Recognition, here going to use "Haar Cascade Classifier" which is already available in the downloaded libraries.
## To get track of libraries,

print(cv2)      


# "C:\Users\square\Anaconda3\Lib\site-packages\cv2" . This is the common path for where the HaarCascade Classifier has been saved.

# Here, Using two cascade classifier i.e., haarcascade_frontalface_default.xml, haarcascade_eye.xml to detect the full face and eyes.

# ###### Building a new folder called Face Recognition and saved it on Desktop, inside that saving both the 'Haar Cascade Classifier' and changing the path of the kernel to that folder.

# In[2]:


## Importing the require basic libraries.

import cv2                  ## for building the face recognition.
import pandas as pd
import numpy as np
import os               ## for changing the directory.

## Changing the path 
os.chdir("C:/Users/square/Desktop/Face_Recognition")

## Importing the Haar Cascade
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detector  = cv2.CascadeClassifier("haarcascade_eye.xml")

## Adding a video capture object
cap = cv2.VideoCapture(0)

while(True):
    ## Capture frame by frame
    ret, frame = cap.read()
    print(ret)
    print(frame)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector.detectMultiScale(gray, scaleFactor = 1.4, minNeighbors = 5, minSize = (30, 30))
    print("Number of Faces", len(faces))
    for (x, y, w, h) in faces:         ## X & y are the location of rectangle.
        print(x, y, w, h)              ## w & h are the width & height.
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
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
            
        ## Saving an Image
        img_item = "Image.jpg"
        cv2.imwrite(img_item, roi_gray)
        
    ## Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
## When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# In[ ]:




