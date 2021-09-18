import cv2
import socket
import time


def detect_faces(cascade, test_image, scaleFactor = 1.1):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 15)

    return image_copy


sock = socket.socket()

host = "192.168.1.159"  # ESP32 IP in local network
port = 80  # ESP32 Server Port
stream_port = 81
#a = ""


tracker_type = "CascadeClassifier"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize some variables
process_this_frame = True
faces = []

version = 1

# Video Stream:
video_source = "Ip Esp32 Stream" # Webcam

# From CAMERA
if (video_source == "Webcam"):

    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()

if (video_source == "Ip Esp32 Stream"):

    cap = cv2.VideoCapture('http://' + host + ':' + str(stream_port) + '/stream')
    ok, frame = cap.read()
    print("Reading video stream from " + 'http://' + host + ':' + str(stream_port) + '/stream' + " - OK")

sock.connect((host, port))
print("Connecting to " + host + ":" + str(port) + " - OK")

haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

x, y, w, h = [0,0,0,0]
count = 0
while True:
    # Read a new frame
    ok, frame = cap.read()
    if (version != 3): gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not ok:
        break

    # Calculate Frames per second (FPS)
    #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    if count==0:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            height, width = frame.shape[:2]

            x, y, w, h = faces[0]

            if(x > 0.5          * width):   sock.send(b"S:3") #print("RIGHT")
            if(x < 0.15         * width):   sock.send(b"S:2") #print("LEFT")
            if(y + h > 0.875    * height):  sock.send(b"S:1") #print("DOWN")
            if(y < 0.0833       * height):  sock.send(b"S:0") #print("UP")
            if(h > 0.6250       * height):  sock.send(b"M:0") #print("BACK")
            if(h < 0.2          * height):  sock.send(b"M:1") #print("FORWARD")

            time.sleep(0.2)

    # Draw bounding box
    img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    count += 1
    if count==8: count=0

    # Display result
    cv2.imshow("Face following", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27: break

sock.close()