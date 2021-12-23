import cv2
import numpy as np
import math

# INITIALIZATION OF CAMERA
cap = cv2.VideoCapture("Video/vid (7).mp4")

# VARIABLE DECLARATION
ball_posX,ball_posy = [],[] # ball center position X and Y
xList = [item for item in range(0,1300)]  # VALUE OF X DIMENSION OF VIDEO/FRAME PIXLES
start = True
prediction = False

while cap.isOpened():
    if start:
        if len(ball_posX) == 10:
            start = False
        ret,frame = cap.read()
        if ret:
            # CROP INTERESTED REGION
            crop_img  = frame[0:900,:]

            # CONVERTING TO HSV COLOR SPACE
            hsv_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

            # DETECTING RED COLOR
            lower_ball_color = np.array([8, 124, 13])
            upper_red_color = np.array([24, 255, 255])

            # MASKING
            mask = cv2.inRange(hsv_img,lower_ball_color,upper_red_color)

            # BITWISE AND OPERATION
            imgcolor = cv2.bitwise_and(crop_img, crop_img, mask=mask)

            # FINDING CONTOURS
            contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

            # sort contours by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for c in contours:
                area = cv2.contourArea(c)
                if area > 500:
                    x,y,w,h = cv2.boundingRect(c)
                    cx,cy = x+(w//2), y+(h//2)
                    ball_posX.append(cx)
                    ball_posy.append(cy)

            if ball_posX:
                # CREATE POLYNOMIAL REGRESSION MODEL FOR PREDICTION ---- y = Ax^2 + Bx + C
                A, B, C = np.polyfit(ball_posX,ball_posy,2)

                for i,(centerx,centery) in enumerate(zip(ball_posX,ball_posy)):
                    center = (centerx,centery)
                    cv2.circle(crop_img, center,10,(147,0,56),-1)
                    """
                    if i == 0:
                        pass
                    else:
                        cv2.line(crop_img , center,(ball_posX[i-1],ball_posy[i-1]),(255,0,0),2)
                    """

                for x in xList:

                    # FIND VALUE OF Y CORRESPONDING TO XLIST

                    y = A*(x**2) + B*x + C

                    cv2.circle(crop_img,(x,int(y)),5,(255,145,255),-1)

                # PREDICTION OF BALL POSITION : x value = 330 to 430 , y value = 590
                # FINDING THE VALUE OF X AND Y

                if len(ball_posX) < 10:

                    a = A
                    b = B
                    c = C-590

                    X = int((-b - math.sqrt(b**2 - 4*a*c))/(2*a))
                    print(X)
                    prediction = 300 <X < 430

                if prediction:
                    text= "Ball will go in Basket"
                    offset = 20
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 2, 2)
                    cv2.rectangle(crop_img,(50-offset,100+offset),(50+offset+w,100-offset-h),(0,255,0),-1)
                    cv2.putText(crop_img,text,(50,100),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2,cv2.LINE_AA)
                else:
                    text = "Ball will not go in Basket"
                    offset = 20
                    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 2, 2)
                    cv2.rectangle(crop_img, (50 - offset, 100 + offset), (50 + offset + w, 100 - offset - h), (0, 0, 255), -1)
                    cv2.putText(crop_img, text, (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('frame',crop_img)

    key = cv2.waitKey(100)

    # START AGAIN AND SHOW RESULT
    if key == ord("s"):
        start = True

cap.release()
cv2.destroyAllWindows()



