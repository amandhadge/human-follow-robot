import RPi.GPIO as GPIO
from time import sleep
from threading import Thread
import imutils
import cv2 as cv
import numpy as np
import time

#Arduino = serial.serial('com87',9600) #Create Serial Port object called arduino Serial Data
#time.sleep(2) #wait for 2 seconds for the communication to get established


lower_yellow=np.array([20,100,100])
upper_yellow=np.array([40,255,255])

GPIO.setmode(GPIO.BOARD)

# Assign GPIO pins for both motors
lm_ena = 33
lm_pos = 35
lm_neg = 37

rm_ena = 36
rm_pos = 38
rm_neg = 40

# Set pins in the output mode
GPIO.setup(lm_ena,GPIO.OUT)
GPIO.setup(lm_pos,GPIO.OUT)
GPIO.setup(lm_neg,GPIO.OUT)

GPIO.setup(rm_ena,GPIO.OUT)
GPIO.setup(rm_pos,GPIO.OUT)
GPIO.setup(rm_neg,GPIO.OUT)

def moveRobot(direction):
    if (direction == "f"):
        print("Forward")
        
        # Left Motor Forward
        GPIO.output(lm_ena,GPIO.HIGH)
        GPIO.output(lm_pos,GPIO.HIGH)
        GPIO.output(lm_neg,GPIO.LOW)

        # Right Motor Forward
        GPIO.output(rm_ena,GPIO.HIGH)
        GPIO.output(rm_pos,GPIO.HIGH)
        GPIO.output(rm_neg,GPIO.LOW)

    
    if (direction == "r"):
        print("Right")
        
        # Left Motor Forward
        GPIO.output(lm_ena,GPIO.HIGH)
        GPIO.output(lm_pos,GPIO.HIGH)
        GPIO.output(lm_neg,GPIO.LOW)

        # Right Motor Backward
        GPIO.output(rm_ena,GPIO.HIGH)
        GPIO.output(rm_pos,GPIO.LOW)
        GPIO.output(rm_neg,GPIO.HIGH)

    if (direction == "l"):
        print("Left")
        
        # Left Motor Backward
        GPIO.output(lm_ena,GPIO.HIGH)
        GPIO.output(lm_pos,GPIO.LOW)
        GPIO.output(lm_neg,GPIO.HIGH)

        # Right Motor Forward
        GPIO.output(rm_ena,GPIO.HIGH)
        GPIO.output(rm_pos,GPIO.HIGH)
        GPIO.output(rm_neg,GPIO.LOW)
        
	
    if (direction == "s"):
        print("Stop")
        
        # Left Motor Backward
        GPIO.output(lm_ena,GPIO.HIGH)
        GPIO.output(lm_pos,GPIO.LOW)
        GPIO.output(lm_neg,GPIO.LOW)

        # Right Motor Forward
        GPIO.output(rm_ena,GPIO.HIGH)
        GPIO.output(rm_pos,GPIO.LOW)
        GPIO.output(rm_neg,GPIO.LOW)        

cam = cv.VideoCapture(0)


while(1):
    ret, frame = cam.read()
    frame = cv.flip(frame,1)

    w = frame.shape[1]
    h = frame.shape[0]
    
    # Smoothen the Image
    image_smooth = cv.GaussianBlur(frame,(7,7),0)

    # Define ROI
    mask = np.zeros_like(frame)

    mask[50:350, 50:350] = [255,255,255]

    image_roi = cv.bitwise_and(image_smooth, mask)
    cv.rectangle(frame,(0,0),(600,400),(0,0,255),2)
    cv.line(frame,(200,0),(200,400),(0,0,255),2)
    cv.line(frame,(400,0),(400,400),(0,0,255),2)
  
    

    # Threshold the Image for Red Color
    image_hsv = cv.cvtColor(image_smooth, cv.COLOR_BGR2HSV)
    image_threshold = cv.inRange(image_hsv, lower_yellow, upper_yellow)

    # Find contours
    _ , contours, _ = cv.findContours(image_threshold,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    # Find the index of the largest contour
    if (len(contours)!=0):
        areas = [cv.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]

        #Pointer on Video
        M = cv.moments(cnt)
        if(M['m00'] != 0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv.circle(frame, (cx, cy), 4, (0,255,0), -1)

            #Cursor Motion
            if cx in range(0,1300):
                if cx < 200:
                    moveRobot('l')
            

                elif cx > 400:
                    moveRobot('r')

                else:
                    moveRobot('f')

        

        
    cv.imshow('Frame', frame)

    key = cv.waitKey(100)
    if key == 27:
        break

camera.release()
cv.destroyAllWindows()
