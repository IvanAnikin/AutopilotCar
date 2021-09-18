import time

import cv2
import sys
import numpy as np
import socket

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


def object_following(testing = False, sensitivity = 0.2, process_every = 5, movement_interval = 1, tracker_num = 0):

    # Set up tracker.
    # Instead of MIL, you can also use

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT', 'MEAN_SHIFT']
    tracker_type = tracker_types[1]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_num)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()



    host = "192.168.1.159"  # ESP32 IP in local network
    port = 80  # ESP32 Server Port
    stream_port = 81

    sock = socket.socket()
    sock.connect((host, port))
    print("Connecting to " + host+ ":" + str(port) + " - OK")

    # Video Stream:
    video_source = "Ip Esp32 Stream" #"Ip Esp32 Stream"  # Webcam

    # From CAMERA
    if (video_source == "Webcam"):
        cap = cv2.VideoCapture(0)
        ok, frame = cap.read()

    if (video_source == "Ip Esp32 Stream"):
        cap = cv2.VideoCapture('http://' + host + ':' + str(stream_port) + '/stream')
        ok, frame = cap.read()
        print("Reading video stream from " + 'http://' + host + ':' + str(stream_port) + '/stream' + " - OK")
    frame_h, frame_w, c = frame.shape


    timestep = 0
    last_movement = time.time()

    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)

    if(tracker_type != "MEAN_SHIFT"):
        # Initialize tracker with first frame and bounding box
        ok = tracker.init(frame, bbox)

    if tracker_type == "MEAN_SHIFT":
        # apply mask
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, np.array((0, 20, 20)), np.array((180, 250, 250)))
        hist_frame = cv2.calcHist([hsv_frame], [0], mask, [180], [0, 180])
        cv2.normalize(hist_frame, hist_frame, 0, 255, cv2.NORM_MINMAX)

        # Setup the termination criteria: 10 iteration 1 pxl movement
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        # Read a new frame
        #ok, frame = cap.read()

        ok, frame = cap.read()
        if not ok:
            break
        #cap.release()

        '''try:
            received = sock.recv(1024).decode("utf-8")  # /512

            if (len(received) < 6):
                distance = received.strip('\r\n')  # line, buffer = received.split('\n', 1)

                if (distance != ''):
        '''

        if(timestep%process_every==0 or timestep==0):

            # Start timer
            timer = cv2.getTickCount()

            if tracker_type == "MEAN_SHIFT":
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv], [0], hist_frame, [0, 180], 1)

                # apply camshift to get the new location
                ret, bbox = cv2.CamShift(dst, bbox, term_crit)

                # Draw a box around the ROI
                pts = cv2.boxPoints(ret)
                pts = np.int0(pts)
                cv2.polylines(frame, [pts], True, 255, 2)

            # Use the q button to quit the operation
            if cv2.waitKey(60) & 0xff == ord('q'):
                break
            else:
                if (tracker_type != "MEAN_SHIFT"):
                    # Update tracker
                    ok, bbox = tracker.update(frame)

                # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                if (tracker_type != "MEAN_SHIFT"):
                    # Draw bounding box
                    if ok:
                        # Tracking success
                        p1 = (int(bbox[0]), int(bbox[1]))
                        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                        #cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                        x,y,w,h = bbox


                        if((time.time()-last_movement)>movement_interval):
                            if(x<frame_w*sensitivity):
                                sock.send(b"H:2")
                                last_movement = time.time()
                            if(x+w>frame_w*(1-sensitivity)):
                                sock.send(b"H:3")
                                last_movement = time.time()

                        #if(x<frame_w*0.2):      cv2.putText(frame, "Left", (100, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2) #sock.send(b"S:2")
                        #if(x+w>frame_w*0.8):    cv2.putText(frame, "Right", (100, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2) #sock.send(b"S:3")

                    else:
                        # Tracking failure
                        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Show object rectangle
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        if (x < frame_w * sensitivity):
            cv2.putText(frame, "Left", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        if(x+w>frame_w*(1-sensitivity)):
            cv2.putText(frame, "Right", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        if(testing):
            # Display tracker type on frame
            cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display bbox on frame
            cv2.putText(frame, "x: " + str(bbox[0]), (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            cv2.putText(frame, "y: " + str(bbox[1]), (100, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            cv2.putText(frame, "box width: " + str(bbox[2]), (100, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            cv2.putText(frame, "box height: " + str(bbox[3]), (100, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display frame size
            cv2.putText(frame, "frame width: " + str(w), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            cv2.putText(frame, "frame height: " + str(h), (100, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display result
        cv2.imshow("Object following", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break

        timestep += 1

        #except socket.error:
        #    pass
