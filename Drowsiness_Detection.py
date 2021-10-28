
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import RPi.GPIO as GPIO
import argparse
import imutils
import time
import dlib
import cv2
import os

alarme_pin = 27
led = 26
def setup():
	GPIO.setmode(GPIO.BCM)
	GPIO.setup(alarme_pin, GPIO.IN)
	GPIO.setup(alarme_pin, GPIO.OUT)
	GPIO.setup(led, GPIO.OUT)

def destroy():
	GPIO.cleanup()

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)


EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
COUNTER = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

camera = VideoStream(usePiCamera=True).start() 
time.sleep(1)
setup()


while True:
	GPIO.output(led,True)	
	image = camera.read()
	image = imutils.resize(image, width=300)	
	flag=0
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	rects = detector(gray, 0)
	for rect in rects:
	        shape = predictor(gray, rect)
	        shape = face_utils.shape_to_np(shape)

      	  	eye = final_ear(shape)
        	ear = eye[0]
	        leftEye = eye [1]
        	rightEye = eye[2]	
        	leftEyeHull = cv2.convexHull(leftEye)
        	rightEyeHull = cv2.convexHull(rightEye)
        	cv2.drawContours(image, [leftEyeHull], -1, (0, 255, 0), 1)
        	cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)
	
	
        	if ear < EYE_AR_THRESH:
        	    COUNTER += 1
	
        	    if COUNTER >= EYE_AR_CONSEC_FRAMES:

        	        cv2.putText(image, "RISCO DE ACIDENTE!!!", (10, 30),
        	                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			GPIO.output(alarme_pin,True)
			time.sleep(1)
	
        	else:
        	   	COUNTER = 0
			GPIO.output(alarme_pin,False)
			time.sleep(1)
	

	cv2.imshow("Frame", image)
    	key = cv2.waitKey(1) & 0xFF
    	if key == ord("q"):
        	break
destroy()
cv2.destroyAllWindows()
camera.stop()
