
import numpy as np
import os
from imageai.Detection import ObjectDetection

from Managers import dataset_manager


datasetManager = dataset_manager.DatasetManager(dataset_type = [1, 1, 1, 1, 1, 1, 1, 1], subname="new", datasets_directory="C:/ML_car/Datasets/Preprocessed/710e9e3")
#datasetManager = dataset_manager.DatasetManager(dataset_type = [1, 0, 0, 0, 0, 1, 0, 1], subname="new_03.06_2", datasets_directory="C:/Users/ivana/OneDrive/Coding/ML/Com_Vis/car_project/Datasets/Dist_normalisation")



detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()

files = [i for i in os.listdir(datasetManager.datasets_directory) if
                 os.path.isfile(os.path.join(datasetManager.datasets_directory, i)) and 'f_s_e_b_c_a_r_d_new' in i]

for file_name in files:

    dataset = np.load(datasetManager.datasets_directory + '/' + file_name, allow_pickle=True)

    for row in dataset:

        frame = row[0] #resized frame - row[1]


