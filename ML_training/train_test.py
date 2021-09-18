
import numpy as np
import os

from ML_training import Train


datasets_directory="C:/ML_car/Datasets/Preprocessed/673243b"
files = [i for i in os.listdir(datasets_directory) if
                 os.path.isfile(os.path.join(datasets_directory, i)) and 'f_s_e_b_c_a_r_d_o' in i]
total_len = 0
for file_name_2 in files: total_len+=len(np.load(datasets_directory + '/' + file_name_2, allow_pickle=True))

count = 0
visualise=True

trainer = Train.Trainer(total_len=total_len, dim=[[0, 0, 1],[1, 1, 1, 1, 1]], model_type="Actor_Critic", batch_size = 32)
#trainer = Train.Trainer(total_len=total_len, dim=[[0, 0, 1],[1, 1, 1, 1, 1]], model_type="DQN", batch_size = 32)

for file_name in files:

    trainer.simulate_on_dataset_2(file_name=file_name, visualise=visualise)
    if visualise: print("\nFinished dataset '{name}' | {count}/{len}".format(name=file_name, count=count+1, len=len(files)))

    count+=1
