import cv2
import socket
import random
import time
import errno


sock = socket.socket()

host = "192.168.1.159"  # ESP32 IP in local network
port = 80  # ESP32 Server Port
stream_port = 81

actions_array = [0, 2, 3]       # 0-Forward; 1-Backward; 2-Left; 3-Right
action_interval = 1
minimal_distance = 15

# Video Stream:
#video_source = "Webcam"
video_source = "Ip Esp32 Stream"

sock.connect((host, port))
print("Connecting to " + host + ":" + str(port) + " - OK")

while True:
    if (video_source == "Webcam"): cap = cv2.VideoCapture(0)
    if (video_source == "Ip Esp32 Stream"): cap = cv2.VideoCapture('http://' + host + ':' + str(stream_port) + '/stream')
    ok, frame = cap.read()
    cap.release()

    try:
        cv2.imshow('frame', frame)
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break
        received = sock.recv(1024).decode("utf-8")  #/512
        if not received:
            cv2.imshow('frame', frame)
            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27: break
            pass
        if(len(received) < 6):
            strip = received.strip('\r\n')  #line, buffer = received.split('\n', 1)
            if(strip != ''):
                print("distance: '" + strip  + "'")

                # SAVE to dataset -- [last_frame, action, distance] --> later change --> [ X={lastframes - classical, edges …}{contours, (detected objects)}, Y=action, reward ]

                # Choose action:    -- now random           --> later from model
                action = random.choice(actions_array)
                if(int(strip) <= minimal_distance):         # if distance smaller than minimal distance
                    action = 2                              # LEFT

                sock.send(b"M:" + str(action).encode())     #send aciton to car
                last_frame = frame

    except socket.error:
        cv2.imshow('frame', frame)
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break
        pass


