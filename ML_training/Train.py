
import numpy as np

import ML_training.Agent as Agents
from Managers import dataset_manager
from tensorflow.keras.optimizers import Adam
import progressbar
from bidict import bidict
import datetime

class Trainer():

    def __init__(
        self,
        dataset_type = [1, 1, 1, 1, 1, 1, 1, 1],
        subname = "new",
        dim=[[0, 0, 0],[0, 0, 1]],      # [0, 1, 1]     #c_dim = (None, 1, 2)   # [[contours, action, distance],[resized, canny_edges, blackAndWhite]]
        vid_dim=[120, 160],
        actions = [0, 2, 3],       # 0-Forward; 1-Backward; 2-Left; 3-Right
        optimizer = Adam(learning_rate=0.01),
        batch_size = 32, #width = 120, height = 160, depth = 2
        load_model=True
        ):
        super().__init__()

        self.actions_relations = bidict({0:0, 1:2, 2:3})    # model:dataset
        self.default_y = 0
        self.output_size = len(actions)
        self.dim = dim[0]
        self.vid_inputs = dim[1]
        self.batch_size = batch_size
        self.vid_dim = np.array(np.append(np.array(vid_dim).astype(int), sum(1 for e in self.vid_inputs if e is not 0))).astype(int)
        self.state_size = [self.dim, self.vid_dim]

        subname = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.datasetManager = dataset_manager.DatasetManager(dataset_type = dataset_type, subname=subname,
                                                             datasets_directory="C:/ML_car/Datasets/Preprocessed/710e9e3") #C:/Users/ivana/OneDrive/Coding/ML/Com_Vis/car_project/Datasets
        self.Agent = Agents.DQN(state_size = self.state_size, actions=actions, optimizer=optimizer, models_directory="C:/ML_car/Models",
                                load_model=load_model, models_names=bidict({"q_name":"DQN_qnetwork"+"_"+subname, "t_name":"DQN_target"+"_"+subname}))

        self.Agent.visualise_model()

    def simulate_on_dataset(self):

        #for dataset in datasets

        bar = progressbar.ProgressBar(maxval=100, #len(self.datasetManager.dataset) / 10,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        model_actions = []

        # Observing state of the environment
        state, action, reward = self.get_state_row_data(row=self.datasetManager.dataset[0])

        for timestep in range(1, len(self.datasetManager.dataset)-1):

            # Calculating action
            model_actions.append(self.Agent.act(state))
            # Taking action
            # -

            # Observing state after taking action
            next_state, action, reward = self.get_state_row_data(row=self.datasetManager.dataset[timestep+1])

            # Storing
            self.Agent.store(state, action, reward, next_state)

            state = next_state

            if len(self.Agent.expirience_replay) > self.batch_size:
                self.Agent.retrain(self.batch_size)

            # Aligning target model if on second last state
            if(timestep==len(self.datasetManager.dataset)-2): self.Agent.alighn_target_model()
            if(timestep%10==0): self.Agent.save_model()

            # Visualisation
            #print("state shape: " + str(np.array(state).shape) + " | action: " + str(action) + " | reward: " + str(reward))
            print('timestep: {timestep}/{dataset_len} | model_actions avg: {model_actions_avg}'.format(
                timestep=timestep, dataset_len=len(self.datasetManager.dataset), model_actions_avg=np.average(model_actions)))

            #print (timestep / len(self.datasetManager.dataset) * 100, " percent complete         \r")

            timestep+=1

        bar.finish()

    def get_state_row_data(self, row):

        num_data = []
        vid_data = []
        state = []
        if (self.dim[0] != 0): num_data.append(row[4])
        if (self.dim[1] != 0): num_data.append(row[5])
        if (self.dim[2] != 0): num_data.append(row[7])
        if (len(num_data) != 0): state.append(num_data)

        if (self.vid_inputs[0] != 0): vid_data.append(row[1])
        if (self.vid_inputs[1] != 0): vid_data.append(row[2])
        if (self.vid_inputs[2] != 0): vid_data.append(row[3])
        if (len(vid_data) != 0): state.append(vid_data)

        state = np.array(state)

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


        '''
        self.Agent.Model.model.fit(
            x=[[np.asarray(self.dataset[:,5]).astype(np.int), np.asarray(self.dataset[:,7]).astype(np.int)],
               [np.asarray(self.dataset[:,2]).astype(object), np.asarray(self.dataset[:,3]).astype(object)]], y=np.asarray(self.dataset[:,5]).astype(np.int),
            epochs=200, batch_size=8)
        '''
        #x = [[np.asarray(self.dataset[:, 4]).astype(np.int), np.asarray(self.dataset[:, 5]).astype(np.int), np.asarray(self.dataset[:, 7]).astype(np.int)],
        # [np.asarray(self.dataset[:, 2]).astype(np.int), np.asarray(self.dataset[:, 3]).astype(np.int)]], y = np.asarray(self.dataset[:, 5]).astype(np.int),
