
import numpy as np
import os
from Managers import dataset_manager
from Managers.preporcess_manager import PreprocessManager


datasetManager = dataset_manager.DatasetManager(dataset_type = [1, 1, 1, 1, 1, 1, 1, 1, 1], subname="cut", datasets_directory="C:/ML_car/Datasets/Preprocessed/710e9e3/with_objects/673243b")
#datasetManager = dataset_manager.DatasetManager(dataset_type = [1, 0, 0, 0, 0, 1, 0, 1], subname="new_03.06_2", datasets_directory="C:/Users/ivana/OneDrive/Coding/ML/Com_Vis/car_project/Datasets/Dist_normalisation")

#datasetManager.preprocess_datasets(new_directory=datasetManager.datasets_directory+"/673243b/", name_base="f_s_e_b_c_a_r_d_o", visualise=True)
datasetManager.visualise_datasets_numbers(name_base="f_s_e_b_c_a_r_d_o")
#datasetManager.visualise_dataset()
#if not os.path.exists(datasetManager.datasets_directory + "/" + datasetManager.dataset_name):
#    np.save(datasetManager.datasets_directory + "/" + datasetManager.dataset_name+"_cut.npy", datasetManager.dataset[6:])
#else: raise FileExistsError('The file already exists')
