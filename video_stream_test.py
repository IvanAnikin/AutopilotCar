import cv2
import serial
import time


video_source = "Ip Esp32 Stream" #" # Webcam

# Video Stream:

# From CAMERA
if (video_source == "Webcam"):

    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()

if (video_source == "Ip Esp32 Stream"):

    cap = cv2.VideoCapture('http://192.168.1.159:81/stream')  # rtsp://
    ok, frame = cap.read()

while True:
    # Read a new frame
    ok, frame = cap.read()

    if not ok:
        break

    # Display result
    cv2.imshow("Stream test", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27: break