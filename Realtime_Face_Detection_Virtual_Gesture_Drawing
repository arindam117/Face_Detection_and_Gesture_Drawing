import cv2
import numpy as np

frameWidth = 640    # 320
frameHeight = 480   # 360

cap = cv2.VideoCapture(0)
facecascade = cv2.CascadeClassifier('frontalface_default.xml')
# 3 is for width and frameWidth = 640, we can directly put 360 as well in place of frameWidth in cap.set()
cap.set(3, frameWidth)
# 4 is for height and we already have made the frameHeight variable = 480
cap.set(4, frameHeight)
# 10 is the id for brightness and 130 is its parameter
cap.set(10, 130)

myColors = [[5,107,0,19,255,255],
            [133,56,0,159,156,255],
            [57,76,0,100,255,255],
            [90,48,0,118,255,255]]
myColorValues = [[51,153,255],          ## BGR
                 [255,0,255],
                 [0,255,0],
                 [255,0,0]]

myPoints =  []  ## [x , y , colorId ]

def findColor(img,myColors,myColorValues):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count = 0
    newPoints=[]
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV,lower,upper)
        x,y=getContours(mask)
        cv2.circle(imgResult,(x,y),15,myColorValues[count],cv2.FILLED)
        if x!=0 and y!=0:
            newPoints.append([x,y,count])
        count +=1
        #cv2.imshow(str(color[0]),mask)
    return newPoints

def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>500:
            #cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x, y, w, h = cv2.boundingRect(approx)
    return x+w//2,y

def drawOnCanvas(myPoints,myColorValues):
    for point in myPoints:
        cv2.circle(imgResult, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED)

while True:

    success, img = cap.read()
    imgResult = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # THis will change the color of the webcam input

    # 1.1 is the scale factor and 4 is the minimum neighbour
    # Here we will put the grayed web cam feed
    faces = facecascade.detectMultiScale(gray, 1.1, 4)
    for x, y, w, h in faces:

            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255, 0), 2)
            objectType = "Face Detected"
            # This will put text in the video feed if there is any face
            # x+(w//2)-50 , y+(h//2)-50 .. these are to shift and adjust the text in the screen
            cv2.putText(img,objectType,(x+(w//2)-50,y+(h//2)-80),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),2)

    newPoints = findColor(img, myColors,myColorValues)
    if len(newPoints)!=0:
        for newP in newPoints:
            myPoints.append(newP)
    if len(myPoints)!=0:
        drawOnCanvas(myPoints,myColorValues)


    cv2.imshow("Result", imgResult)
    cv2.imshow('video_output', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
