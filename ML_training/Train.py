import time

import numpy as np
import sys
from bidict import bidict

from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

import ML_training.Agent as Agents
from Managers import dataset_manager

class Trainer():

    def __init__(
        self,
        dataset_type = [1, 1, 1, 1, 1, 1, 1, 1],
        subname = "new",
        datasets_directory = "C:/ML_car/Datasets/Preprocessed/710e9e3",
        dim=[[0, 0, 0],[0, 1, 1]],      # [0, 1, 1]     #c_dim = (None, 1, 2)   # [[contours-not works, action, distance]-not works,[resized-not works, canny_edges, blackAndWhite]]
        model_subname = "",
        vid_dim=[120, 160],
        actions = [0, 2, 3],       # 0-Forward; 1-Backward; 2-Left; 3-Right
        optimizer = Adam(learning_rate=0.01),
        batch_size = 32, #width = 120, height = 160, depth = 2
        load_model=False
        ):
        super().__init__()

        if model_subname=="": model_subname = str(dim)
        self.batch_size = batch_size
        self.actions_relations = bidict({0:0, 1:2, 2:3})    # model:dataset
        self.default_y = 0
        self.output_size = len(actions)
        self.dim = dim[0]
        self.vid_inputs = dim[1]
        self.vid_dim = np.array(np.append(np.array(vid_dim).astype(int), sum(1 for e in self.vid_inputs if e is not 0))).astype(int)
        self.state_size = [self.dim, self.vid_dim]
        self.process_string = '\r timestep: {timestep}/{dataset_len} - {percentage}% | model_actions avg: {model_actions_avg} | total time: {total_time} | ' \
                              'average step time: {step_time} | remaining time: {time_left}'

        #model_subname += "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        q_name = "DQN_qnetwork" + "_" + model_subname
        t_name = "DQN_target" + "_" + model_subname

        self.datasetManager = dataset_manager.DatasetManager(dataset_type = dataset_type, subname=subname,
                                                             datasets_directory=datasets_directory) #C:/Users/ivana/OneDrive/Coding/ML/Com_Vis/car_project/Datasets
        self.Agent = Agents.DQN(state_size = self.state_size, actions=actions, optimizer=optimizer, models_directory="C:/ML_car/Models",
                                load_model=load_model, models_names=bidict({"q_name":q_name, "t_name":t_name}))

        self.Agent.visualise_model()

    def simulate_on_dataset(self, file_name=""):

        if file_name=="": dataset=self.datasetManager.dataset
        else: dataset = np.load(self.datasetManager.datasets_directory + '/' + file_name, allow_pickle=True)
        #for dataset in datasets

        model_actions = []
        start = time.time()

        # Observing state of the environment
        state, action, reward = self.get_state_row_data(row=dataset[0])

        for timestep in range(1, len(dataset)-1):

            # Calculating action
            model_actions.append(self.Agent.act(state))
            # Taking action
            # -

            # Observing state after taking action
            next_state, action, reward = self.get_state_row_data(row=dataset[timestep+1])

            # Storing
            self.Agent.store(state, action, reward, next_state)

            state = next_state

            if len(self.Agent.expirience_replay) > self.batch_size:
                self.Agent.retrain(self.batch_size)

            # Aligning target model if on second last state
            if(timestep==len(dataset)-2): self.Agent.alighn_target_model()
            if(timestep%10==0): self.Agent.save_model(subname=file_name)

            # Visualisation
            sys.stdout.write(self.process_string.format(
                timestep=timestep, dataset_len=len(dataset), percentage=round(timestep/len(dataset)*100, 2),
                model_actions_avg=round(np.average(model_actions),2), total_time=round(time.time()-start,2),
                step_time=round((time.time()-start)/timestep,2), time_left=time.strftime('%H:%M:%S', time.gmtime((time.time()-start)/timestep*(len(dataset)-timestep)))))
            sys.stdout.flush()
            if(timestep%50==0): print()

            timestep+=1

    def get_state_row_data(self, row):

        num_data = []
        vid_data = []
        state = []
        if (self.dim[0] != 0): num_data.append(row[4][0]) #np.array #tf.convert_to_tensor(row[4][0], dtype=tf.int64) #np.asarray(row[4][0]).astype(np.float32)
        if (self.dim[1] != 0): num_data.append(row[5])
        if (self.dim[2] != 0): num_data.append(row[7])
        if (len(num_data) != 0): state.append(num_data)

        if (self.vid_inputs[0] != 0): vid_data.append(row[1])
        if (self.vid_inputs[1] != 0): vid_data.append(row[2])
        if (self.vid_inputs[2] != 0): vid_data.append(row[3])
        if(len(num_data)!=0):vid_data = [vid_data]
        if (len(vid_data) != 0): state.append(vid_data)

        state = np.array(state)
        #state2 = [state1[0], [state1[1]]]

        # Convert action to model action using bidict
        action = self.actions_relations.inverse[row[5]]
        reward = row[6]

        return state, action, reward

    def test(self):
        num_data=[]
        vid_data=[]
        x=[]
        if(self.dim[0] != 0): num_data.append(self.datasetManager.dataset[:, 4])
        if(self.dim[1] != 0): num_data.append(self.datasetManager.dataset[:, 5])
        if(self.dim[2] != 0): num_data.append(self.datasetManager.dataset[:, 7])
        if(len(num_data) != 0): x.append(num_data)

        if(self.vid_inputs[0] != 0): vid_data.append(self.datasetManager.dataset[:, 1])
        if(self.vid_inputs[1] != 0): vid_data.append(self.datasetManager.dataset[:, 2])
        if(self.vid_inputs[2] != 0): vid_data.append(self.datasetManager.dataset[:, 3])
        if(len(vid_data) != 0): x.append(vid_data)

        y = np.array([len(self.datasetManager.dataset), self.output_size]).fill(0)
        #y[]
        #self.Agent.Model.model.fit(x=x, y=[np.asarray(self.datasetManager.dataset[:,5]).astype(int)], epochs=200, batch_size=8) #np.int
