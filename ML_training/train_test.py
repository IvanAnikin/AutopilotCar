
import numpy as np
import os

from ML_training import Train

trainer = Train.Trainer()

datasets_directory="C:/ML_car/Datasets/Preprocessed/710e9e3"
files = [i for i in os.listdir(datasets_directory) if
                 os.path.isfile(os.path.join(datasets_directory, i)) and 'f_s_e_b_c_a_r_d_new' in i]
for file_name in files:

    trainer.simulate_on_dataset(file_name=file_name)
