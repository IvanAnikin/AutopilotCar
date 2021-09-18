import cv2
import numpy as np
import socket
import random
import datetime

import Managers.dataset_manager as dataset_manager
import Managers.preporcess_manager as preprocess_manager


actions_array = [0, 2, 3]       # 0-Forward; 1-Backward; 2-Left; 3-Right
action_interval = 1
minimal_distance = 20
contours_count = 4

type = "explorer"

preprocess = False
visualise = True
dataset_type = [1, 1, 1, 1, 1, 1, 1, 1, 1]
#if(preprocess):
#    dataset_type = [1, 0, 1, 0, 1, 1, 1, 1] # - frame, canny, edges, action, reward, distance
#else:
#    dataset_type = [1, 0, 0, 0, 0, 1, 0, 1]  # - frame, action, distance

data = []   # later --> LOAD FROM PREVIOUS DATASET
subname = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") #_%H:%M:%S
print(subname)
datasetManager = dataset_manager.DatasetManager(dataset_type=dataset_type, subname=subname, data=data, save_every = 5, minimal_distance = minimal_distance, type=type)
preprocessManager = preprocess_manager.PreprocessManager(contours_count = contours_count)


# Video Stream:
#video_source = "Webcam"
video_source = "Ip Esp32 Stream"

sock = socket.socket()
host = "192.168.1.159"  # ESP32 IP in local network
port = 80  # ESP32 Server Port
stream_port = 81

sock.connect((host, port))
print("Connecting to " + host + ":" + str(port) + " - OK")

#count = 0
while True:
    if (video_source == "Webcam"): cap = cv2.VideoCapture(0)
    if (video_source == "Ip Esp32 Stream"): cap = cv2.VideoCapture('http://' + host + ':' + str(stream_port) + '/stream')

    ok, frame = cap.read()
    if(not("action" in locals())): last_frame = frame.copy()
    cap.release()

    try:
        received = sock.recv(1024).decode("utf-8")  #/512
        #cv2.imshow('frame', frame)
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break

        if(len(received) < 6):

            distance = received.strip('\r\n')  #line, buffer = received.split('\n', 1)
            if(distance != ''):
                dataset = []
                #row = np.empty([len(dataset_type)])
                row = []
                canny_edges = []
                print("distance: '" + str(distance)  + "'")
                distance = int(distance)

                if(dataset_type[1]): resized, dim = preprocessManager.resized(frame=frame)
                if (dataset_type[3]):
                    blackAndWhite = datasetManager.preprocessManager.blackAndWhite(frame=frame)
                if(dataset_type[4]):
                    canny_edges, contours = preprocessManager.canny_and_contours(frame=frame)
                    if (visualise): datasetManager.visualise_contours(frame=frame, contours=contours, canny_edges=canny_edges)
                elif(dataset_type[2]):
                    canny_edges = preprocessManager.canny_edges(frame=frame)
                    cv2.imshow('canny_edges', canny_edges)
                objects = []
                if type == "explorer": objects = datasetManager.preprocessManager.objects_detection(frame=frame)

                if(dataset_type[6] and "last_frame" in locals()): reward = preprocessManager.reward_calculator(frame=frame, last_frame=last_frame, distance=distance, canny_edges=canny_edges)
                # SAVE to dataset --> later change --> [ X={lastframes - classical, edges …}{contours, (detected objects)}, Y=action, reward ]
                if("action" in locals()):
                    count = 0
                    for dataset_type_row in dataset_type:
                        if(dataset_type_row): row.append(globals()[datasetManager.dataset_vars[count]]) #row[count] = globals()[datasetManager.dataset_vars[count]]
                        else: row.append(None)
                        count += 1
                    dataset.append(row)
                    datasetManager.save_data(np.array(dataset))

                # Choose action:    -- now random           --> later from model
                action = random.choice(actions_array)
                if(int(distance) <= minimal_distance):      # if distance smaller than minimal distance
                    action = 2                              # LEFT

                sock.send(b"M:" + str(action).encode())     #send aciton to car
                last_frame = frame

                if (visualise):
                    with_text = frame.copy()
                    cv2.putText(with_text, 'Action: ' + str(action), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(with_text, 'Distance: ' + str(distance), (10, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                    if(preprocess): cv2.putText(with_text, 'Reward: ' + str(reward), (10, 150), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.imshow('frame', with_text)

                    k = cv2.waitKey(30) & 0xff
                    if k == 27:
                        break

    except socket.error:
        pass

