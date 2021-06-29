import numpy
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

from Managers import preporcess_manager

class DatasetManager():
    def __init__(self, dataset_type = [1, 1, 1, 1, 1, 1, 1, 1, 1], subname="", data=[], save_every = 5, step_from_saving = 0, minimal_distance = 15,
                 type = "explorer", datasets_directory="C:/ML_car/Datasets/Preprocessed/710e9e3"):   #"../Datasets" #/Normalised brightness
        self.save_every = save_every
        self.step_from_saving = step_from_saving
        self.dataset_name_letters = ["f", "s", "e", "b", "c", "a", "r", "d", "o"]
        self.dataset_vars = ["last_frame", "resized", "canny_edges", "blackAndWhite", "contours", "action", "reward", "distance", "Detected objects"]
        self.dataset_type = dataset_type
        self.datasets_directory = datasets_directory
        self.dataset_name = self.dataset_name_from_type(dataset_type, subname=subname)
        self.dataset_name_full = self.datasets_directory + '/' + self.dataset_name + '.npy'
        if os.path.exists(self.dataset_name_full): self.dataset = np.load(self.dataset_name_full, allow_pickle=True)
        self.data = data
        self.minimal_distance = minimal_distance
        self.frame_position = 0
        self.action = 1
        self.distance = 2
        self.reward = 3
        self.type = type
        self.preprocessManager = preporcess_manager.PreprocessManager(contours_count = 5, minimal_distance = minimal_distance, type = type)
        self.datasets_vis_string = "{4} \n Actions: \n \t average: {0} \n \t sum: {1} \n Distances: \n \t average: {2} \n \t sum: {3} \n Len: {5} \n"

    def save_data(self, data):

        self.data.append(data)

        self.step_from_saving += 1
        if(self.step_from_saving == self.save_every):
            np.save(self.datasets_directory + '/' + self.dataset_name, self.data)
            self.step_from_saving = 0

    def dataset_name_from_type(self, type=[1, 0, 0, 0, 0, 1, 0, 1], subname = "1"):

        dataset_name = ""
        count = 0
        for type_element in type:
            if(type_element):
                if(count): dataset_name += "_"
                dataset_name += self.dataset_name_letters[count]
            count += 1

        dataset_name += ("_" + subname)

        return dataset_name

    # Frames conversion and reward calculation
    def dataset_preprocess(self, dataset = []):

        if(dataset == []): dataset = np.load(self.dataset_name_full, allow_pickle=True)
        new_dataset = []
        count = 0
        for row in dataset:
            last_frame = row[0]
            action = row[5]
            distance = row[7]
            objects = row[8]
            resized, dim = self.preprocessManager.resized(last_frame)

            # create canny frame and contours
            canny_edges, contours = self.preprocessManager.canny_and_contours(frame=last_frame)
            canny_edges = cv2.resize(canny_edges, dim, interpolation=cv2.INTER_AREA)

            # gray and black and wite frames
            grayImage = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

            # calculating the reward
            reward = self.preprocessManager.reward_calculator(frame=last_frame,
                                                               last_frame=dataset[count - 1][self.frame_position],
                                                               distance=distance, canny_edges=canny_edges, objects=objects)

            new_row = np.array([last_frame, resized, canny_edges, blackAndWhiteImage, contours, action, reward, distance, objects])
            new_dataset.append(new_row)
            count += 1

        print(np.array(new_dataset).shape)
        return np.array(new_dataset)

    def preprocess_datasets(self, new_directory="./Preprocessed/", name_base="f_s_e_b_c_a_r_d_o", visualise = False):

        files = [i for i in os.listdir(self.datasets_directory) if
                 os.path.isfile(os.path.join(self.datasets_directory, i)) and name_base in i]
        for file_name in files:
            dataset = np.load(self.datasets_directory + '/' + file_name, allow_pickle=True)
            self.preprocessManager.detected_objects = []

            #if not os.path.exists(new_directory + file_name): np.save(new_directory + file_name, self.dataset_preprocess(dataset=dataset))
            #else: raise FileExistsError('The file already exists')
            new = self.dataset_preprocess(dataset=dataset)
            if visualise: print("detected_objects: {objects} | reward avg: {avg}".format(objects = self.preprocessManager.detected_objects, avg=np.round(np.average(new[:,6]),2)))
            print()


    def visualise_dataset_numbers(self):

        dataset = np.load(self.datasets_directory + '/' + self.dataset_name + '.npy', allow_pickle=True)
        rewards = []

        for row in dataset:
            rewards.append(row[6])

        print("Average reward: ", np.average(rewards))
        contours = row[4]
        print("One row contour shape: ", dataset[0][4].shape())

    def visualise_contours(self, frame, contours, canny_edges):

        with_contours = frame.copy()
        for c in contours:
            rows, cols = canny_edges.shape[:2]
            [vx, vy, x, y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
            lefty = int((-x * vy / vx) + y)
            righty = int(((cols - x) * vy / vx) + y)
            try:
                cv2.line(with_contours, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
            except:
                pass
        cv2.imshow('canny_edges', canny_edges)
        cv2.imshow('with_contours', with_contours)

    def visualise_dataset(self, dataset = []):

        if dataset==[]: dataset = np.load(self.dataset_name_full, allow_pickle=True)
        count = 0
        for row in dataset:

            if(self.dataset_type[0]): frame = row[0]
            elif(self.dataset_type[1]):frame = row[1]
            elif(self.dataset_type[2]):frame = row[2]
            elif(self.dataset_type[3]):frame = row[3]
            if (self.dataset_type[1]):
                if (self.dataset_type[8]):
                    for id,box in row[8]:
                        # extract the bounding box coordinates
                        (x, y) = (box[0], box[1])
                        (w, h) = (box[2], box[3])
                        # draw a bounding box rectangle and label on the frame
                        color = [int(c) for c in self.preprocessManager.COLORS[id]]
                        cv2.rectangle(row[1], (x, y), (x + w, y + h), color, 2)
                        text = "{}".format(self.preprocessManager.LABELS[id])
                        cv2.putText(row[1], text, (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.imshow('resized', row[1])
            if (self.dataset_type[2]): cv2.imshow('canny_edges', row[2])
            if (self.dataset_type[3]): cv2.imshow('blackAndWhite', row[3])
            if(self.dataset_type[4]):
                try:
                    with_contours = frame.copy()
                    for c in row[4]:
                        rows, cols = frame.shape[:2]
                        [vx, vy, x, y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
                        lefty = int((-x * vy / vx) + y)
                        righty = int(((cols - x) * vy / vx) + y)
                        cv2.line(with_contours, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
                    cv2.imshow('with_contours', with_contours)
                except:
                    pass
            if("frame" in locals()):
                if(self.dataset_type[5]): cv2.putText(frame, 'Action: ' + str(row[5]), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                if(self.dataset_type[7]): cv2.putText(frame, 'Distance: ' + str(row[7]), (10, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                if(self.dataset_type[6]): cv2.putText(frame, 'Reward: ' + str(row[6]), (10, 150), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, 'Count: ' + str(count), (10, 200), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow('frame,', row[0])
            count+=1

            k = cv2.waitKey(500) & 0xff
            if k == 27:
                break

    def visualise_datasets(self):
        files = [i for i in os.listdir(self.datasets_directory) if
                 os.path.isfile(os.path.join(self.datasets_directory, i)) and 'f_a_d_new' in i]
        for file_name in files:
            dataset = np.load(self.datasets_directory + '/' + file_name, allow_pickle=True)

            self.visualise_dataset(dataset)

    def convert_old_dataset(self):

        dataset = np.load(self.datasets_directory + '/' + self.dataset_name + '.npy', allow_pickle=True)
        new_dataset = []

        for row in dataset:
            new_dataset.append(np.array([row[0], None, None, None, None, row[5], None, row[7]]))

        return new_dataset

    def to_excel(self):

        files = [i for i in os.listdir(self.datasets_directory) if os.path.isfile(os.path.join(self.datasets_directory, i)) and 'f_a_d_new' in i]
        for file_name in files:
            dataset = np.load(self.datasets_directory + '/' + file_name, allow_pickle=True)
            data_df = pd.DataFrame(dataset) #Convert ndarray format to DataFrame

            # Change the index of the table
            data_df.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']    # Columns names
            #data_df.index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']      # Rows names

            # Write the file into the excel table
            writer = pd.ExcelWriter('hhh.xlsx')  # Key 2, create an excel sheet named hhh
            data_df.to_excel(writer, 'page_1',
                             float_format='%.5f')  # Key 3, float_format controls the accuracy, write data_df to the first page of the hhh form. If there are multiple files, you can write in page_2
            writer.save()  # Key 4

    def visualise_charts(self, chart=True):
        if os.path.isfile('./' + "A&D_stats" + ".npy"):
            stats=np.load('./' + "A&D_stats" + ".npy")
            files = [i for i in os.listdir(self.datasets_directory) if
                     os.path.isfile(os.path.join(self.datasets_directory, i)) and 'f_a_d_new' in i]
            for file_name in files:
                dataset = np.load(self.datasets_directory + '/' + file_name, allow_pickle=True)
                self.plot_dataset(dataset=dataset, name=file_name, chart=chart)
        else:
            stats = []
            files = [i for i in os.listdir(self.datasets_directory) if os.path.isfile(os.path.join(self.datasets_directory, i)) and 'f_a_d_new' in i]
            for file_name in files:
                #file_name = "f_a_d_new.npy"
                dataset = np.load(self.datasets_directory + '/' + file_name, allow_pickle=True)

                actions_avg, distances_avg = self.plot_dataset(dataset=dataset, name=file_name, chart=chart)
                stats.append([file_name, actions_avg, distances_avg])

            np.save("A&D_stats",stats)

        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle("Avg action and distance")
        #print(stats[:,1])
        ax1.plot(stats[:,0],
                 np.array(stats[:,1]).astype(float))
        ax2.plot(stats[:,0], np.array(stats[:,2].astype(float)))
        plt.show()

    def plot_dataset(self, dataset, name, chart=True):

        x = []
        for i in range(len(dataset)): x.append(i)
        x = np.array(x)
        actions = np.array(dataset[:, 5]).astype(int)
        distances = np.array(dataset[:, 7]).astype(int)

        if(chart):
            fig, (ax1, ax2) = plt.subplots(2)
            fig.suptitle(name)
            ax1.plot(x, actions)
            ax2.plot(x, distances)
            plt.show()

        print(self.datasets_vis_string.format(np.average(actions), np.sum(actions), np.average(distances),
                                              np.sum(distances), name, len(dataset)))

        return np.average(actions), np.average(distances)

    def clean_datasets(self, visualise=False, dist_coefficient=4, plot=True, chart=True, save=False, path="./"):

        files = [i for i in os.listdir(self.datasets_directory) if
                 os.path.isfile(os.path.join(self.datasets_directory, i)) and 'f_a_d_new' in i]
        for file_name in files:
            self.dataset = dataset = np.load(self.datasets_directory + '/' + file_name, allow_pickle=True)

            if plot: self.plot_dataset(dataset=dataset, name=file_name, chart=chart)

            if visualise: self.plot_dataset(dataset=dataset, name=file_name)

            count = 0
            for row in dataset:
                # Distance <= 0 ==> average distance in range
                if row[7] <= 0: row[7] = self.get_range_avg(var_range=5, param=7, i=count)
                # Distance > coefficient*avg ==> average distance in range
                if row[7] > dist_coefficient * np.average(np.array(self.dataset[:, 7]).astype(int)): row[7] = self.get_range_avg(var_range=5, param=7, i=count)

                #Video brightness normalisation
                frame = row[0]
                if (visualise): cv2.imshow('frame', frame)

                cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
                row[0] = frame

                if (visualise):
                    cv2.imshow('normalised', frame)
                    if cv2.waitKey(500) & 0xFF == ord('q'):
                        break
                #//Video brightness normalisation
                count += 1

            if (visualise): cv2.destroyAllWindows()
            if plot: self.plot_dataset(dataset=dataset, name=file_name+" cleaned", chart=chart)

            if save: np.save(path + "/" + file_name, dataset)

    def video_brighness_normalisation(self, visualise=False, dataset=[]):

        if dataset==[]: dataset = np.load(self.dataset_name_full, allow_pickle=True)
        new_dataset = dataset
        count = 0
        for row in dataset:
            frame = row[0]
            if(visualise): cv2.imshow('frame', frame)

            cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
            new_dataset[count][0] = frame

            if(visualise):
                cv2.imshow('normalised', frame)
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            count += 1
        if(visualise): cv2.destroyAllWindows()

        return new_dataset

    def get_range_avg(self,var_range=5, param=7, i=0):
        prev = next = np.average(np.array(self.dataset[:, param]).astype(int))
        for j in range(var_range):
            if i - j >= 0:
                if (self.dataset[i - j][param] != 0):
                    prev = self.dataset[i - j][param]
            if (i + j < len(self.dataset)):
                if (self.dataset[i + j][param] != 0):
                    next = self.dataset[i + j][param]
        return (prev+next)/2

    def remove_zeros_less(self, var_range=5, param=7):

        i = 0
        for row in self.dataset:
            if row[param] <= 0: row[param] = self.get_range_avg(var_range=var_range, param=param, i=i)
            i+=1

    def remove_too_high(self, param=7, coefficient=7):

        i=0
        for row in self.dataset:
            if row[param] > coefficient * np.average(self.dataset[:,param]): row[param] = self.get_range_avg(param=param, i=i)
            i+=1

    def add_objects(self):

        files = [i for i in os.listdir(self.datasets_directory) if
                 os.path.isfile(os.path.join(self.datasets_directory, i)) and 'f_s_e_b_c_a_r_d_new' in i]


        for file_name in files:

            dataset = np.load(self.datasets_directory + '/' + file_name, allow_pickle=True)
            new_dataset = []

            print("Dataset: {name}".format(name=file_name))

            count = 0
            for row in dataset:
                objects_detected = self.preprocessManager.objects_detection(row[0], visualise=False, scale_percent=50)

                last_frame, resized, canny_edges, blackAndWhiteImage, contours, action, reward, distance = row

                new_row = (last_frame, resized, canny_edges, blackAndWhiteImage, contours, action, reward, distance, objects_detected)
                new_dataset.append(new_row)

                print("Step: {step}/{len} | Objects detected: {objects_detected}".format(objects_detected=objects_detected, step=count, len=len(dataset)))

                count+=1

            print("New dataset shape: {shape}".format(shape = np.array(new_dataset).shape))
            if not os.path.exists(self.datasets_directory + "/with_objects/" + file_name): np.save(self.datasets_directory + "/with_objects/" + file_name, np.array(new_dataset))
            else: raise FileExistsError('The file already exists')

# Dataset type:

# f_r_c_e_g_t_b_a_r_d_o
# 0 - frame		    - f     - not use   for training
# 1 - resized		- s     - (use)     for training    - doesn't work
# 2 - canny_edges	- e     - (use)     for training    - works
# 3 - blackAndWhite	- b     - use       for training    - works
# 4 - contours		- c     - use       for training    - doesn't work
# 5 - action		- a     - use       for training    - doesn't work
# 6 - reward		- r     - use       for training    - doesn't work
# 7 - distance		- d     - (use)     for training    - works
# 8 - objects       - o     - (use)     for training

# Dataset versions:

# 710e9e3   -- lower than min distance punishment, frames difference if edges exist, reward for count of canny edges
#           -- + reward for new object
