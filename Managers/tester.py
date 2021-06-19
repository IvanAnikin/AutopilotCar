
import numpy as np
import os
import matplotlib.pyplot as plt

from Managers import dataset_manager


datasetManager = dataset_manager.DatasetManager(dataset_type = [1, 1, 1, 1, 1, 1, 1, 1], subname="new", datasets_directory="C:/Users/ivana/OneDrive/Coding/ML/Com_Vis/car_project/Datasets/Preprocessed/710e9e3")
#datasetManager = dataset_manager.DatasetManager(dataset_type = [1, 0, 0, 0, 0, 1, 0, 1], subname="new_03.06_2", datasets_directory="C:/Users/ivana/OneDrive/Coding/ML/Com_Vis/car_project/Datasets/Dist_normalisation")

rewards = []
files = [i for i in os.listdir(datasetManager.datasets_directory) if
                 os.path.isfile(os.path.join(datasetManager.datasets_directory, i)) and 'f_a_d_new' in i]
for file_name in files:
    datasetManager.dataset = dataset = np.load(datasetManager.datasets_directory + '/' + file_name, allow_pickle=True)
    for row in dataset: rewards.append(row[6])
rewards = np.array(rewards)
print(rewards.shape)
print(np.average(rewards))
x = []
for i in range(len(rewards)): x.append(i)
x = np.array(x)
plt.plot(x, rewards)
plt.show()
