
import numpy as np
import os
from Managers import dataset_manager
from Managers.preporcess_manager import PreprocessManager


datasetManager = dataset_manager.DatasetManager(dataset_type = [1, 1, 1, 1, 1, 1, 1, 1, 1], subname="new", datasets_directory="C:/ML_car/Datasets/Preprocessed/710e9e3/with_objects")
#datasetManager = dataset_manager.DatasetManager(dataset_type = [1, 0, 0, 0, 0, 1, 0, 1], subname="new_03.06_2", datasets_directory="C:/Users/ivana/OneDrive/Coding/ML/Com_Vis/car_project/Datasets/Dist_normalisation")

datasetManager.preprocess_datasets(new_directory=datasetManager.datasets_directory+"/a4fc1fd/", name_base="f_s_e_b_c_a_r_d_o", visualise=True)
