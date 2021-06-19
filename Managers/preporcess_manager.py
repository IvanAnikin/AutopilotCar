import cv2
import numpy as np

class PreprocessManager():
    def __init__(self, contours_count = 5, minimal_distance = 15, type = "explorer",
                 edges_avg = 120000, edges_max = 500000, edges_coefficient = 20000, min_edges_sum_for_difference = 100):
        self.contours_count = contours_count
        self.minimal_distance = minimal_distance
        self.type = type
        self.edges_avg = edges_avg
        self.edges_max = edges_max
        self.edges_coefficient = edges_coefficient
        self.min_edges_sum_for_difference = min_edges_sum_for_difference

    def canny_edges(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # grayscale frame
        return(cv2.Canny(frame_gray, 50, 100, 3))               # canny edges detection\

    def resized(self, frame):
        # reduce image size
        scale_percent = 50  # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        return resized, dim

    def canny_and_contours(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canny_edges = cv2.Canny(frame_gray, 50, 100, 3)

        ret, threshold = cv2.threshold(canny_edges, 170, 255, cv2.THRESH_BINARY)
        im, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # contours detection on thresholded cropped frame
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))

        chosen_contours = []
        count = 0
        for c in cntsSorted:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, False)
            if area < 10001 and 100 < perimeter < 1000:
                if (count < self.contours_count):
                    chosen_contours.append(c)
                    count += 1

        return canny_edges, chosen_contours

    # Dependencies:
    #   common: distance    - too close = -10                                                                   # -> Other proportions of dependencies
    #   explorer: frame difference from others
    def reward_calculator(self, frame, last_frame, distance, canny_edges=[]):

        reward = 0
        if (int(distance) <= self.minimal_distance): reward = reward - 10
        elif((self.type == "explorer" or self.type == "detective") and len(canny_edges) > 0):                   # reward for difference only if not too close distance
            edges_sum = 0
            for edges_row in canny_edges:
                for edge in edges_row:
                    edges_sum += edge
            # calculate frame difference from the previous frame if not too small edges sum                # -> difference from some ammount of the PREVIOUS/  (previous+next - unable for realtime training)
            if(edges_sum > self.min_edges_sum_for_difference):
                difference = cv2.absdiff(frame, last_frame).astype(np.uint8)
                difference = round(100 - (np.count_nonzero(difference) * 100) / difference.size, 2)  # 0 to 15
                reward += difference

            # Calculating edges count for higher reward for more edges
            edges_reward = edges_sum/self.edges_coefficient - self.edges_avg/self.edges_coefficient

            reward += edges_reward  # -> difference/different variable

        return reward

