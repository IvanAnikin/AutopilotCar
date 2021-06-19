import cv2
import numpy as np

good = []

class object_detection():

    def __init__(self, name = 'helmet'):
        self.name = name
        self.good = []
        self.keypoints_1 = []
        self.keypoints_2 = []
        self.images_directory = './src/img/'
        self.ending = '.jpg'
        self.image_template = cv2.imread(self.images_directory + self.name + self.ending)



def finding_object(new_image, MIN_MATCH_COUNT=15, count=0, object = object_detection()):

    if(count == 0):
        # Function that compares input image to template
        # It then returns the number of SIFT matches between them
        image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(object.image_template, cv2.COLOR_BGR2GRAY)

        # Create SIFT detector object
        #sift = cv2.SIFT()
        sift = cv2.xfeatures2d.SIFT_create()
        # Obtain the keypoints and descriptors using SIFT
        object.keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
        object.keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)

        # Define parameters for our Flann Matcher
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 3)
        search_params = dict(checks = 100)

        # Create the Flann Matcher object
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Obtain matches using K-Nearest Neighbor Method
        # the result 'matchs' is the number of similar matches found in both images
        matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

        object.good = []
        # store all the good matches as per Lowe's ratio test.
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                object.good.append(m)

    object_found = False
    new_frame = new_image.copy()

    if len(object.good) > MIN_MATCH_COUNT:
        src_pts = np.float32([object.keypoints_1[m.queryIdx].pt for m in object.good]).reshape(-1, 1, 2)
        dst_pts = np.float32([object.keypoints_2[m.trainIdx].pt for m in object.good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        new_frame = cv2.drawMatches(new_frame, object.keypoints_1, object.image_template, object.keypoints_2, object.good, None, **draw_params)

        object_found = True

    return new_frame, object_found, object


host = "192.168.1.159"  # ESP32 IP in local network
port = 80  # ESP32 Server Port
stream_port = 81

# Video Stream:
video_source = "Webcam" #"Ip Esp32 Stream" # Webcam

# From CAMERA
if (video_source == "Webcam"):

    cap = cv2.VideoCapture(1)
    ok, frame = cap.read()

if (video_source == "Ip Esp32 Stream"):

    cap = cv2.VideoCapture('http://' + host + ':' + str(stream_port) + '/stream')
    ok, frame = cap.read()
    print("Reading video stream from " + 'http://' + host + ':' + str(stream_port) + '/stream' + " - OK")

count = 0
object_found = False
object = object_detection(name='helmet')
while True:

    ret, frame = cap.read()

    height, width = frame.shape[:2]

    # Get number of SIFT matches
    new_frame, object_found, object = finding_object(frame, 15, count, object)
    if object_found:
        cv2.putText(new_frame ,'Object Found' ,(50 ,50), cv2.FONT_HERSHEY_COMPLEX, 2 ,(0, 255, 0), 2)

    cv2.imshow('Object Detector using SIFT', new_frame)

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

    count +=1
    if(count==10): count=0

cap.release()
cv2.destroyAllWindows()