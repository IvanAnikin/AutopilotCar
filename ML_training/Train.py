import time

import numpy as np
import sys
from bidict import bidict
import time

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
        datasets_directory = "C:/ML_car/Datasets/Preprocessed/673243b",
        dim=[[0, 0, 1],[1, 1, 1, 1, 1]],      #[1, 1, 1, 1, 1] [0, 0, 0, 1, 1]     #c_dim = (None, 1, 2)   # [[contours-not works, action, distance],[resized, canny_edges, blackAndWhite]]
        model_subname = "",
        vid_dim=[120, 160],
        actions = [0, 2, 3],       # 0-Forward; 1-Backward; 2-Left; 3-Right
        optimizer = Adam(learning_rate=0.01),
        batch_size = 32, #width = 120, height = 160, depth = 2
        load_model=True,
        total_len=0,
        type="explorer",
        model_type="DQN"
        ):
        super().__init__()

        if model_subname=="": model_subname = str(dim)
        self.model_type=model_type
        self.avg_timestep=0
        self.type = type
        self.model_subname = model_subname
        self.total_len = total_len
        self.total_timesteps=0
        self.batch_size = batch_size
        self.actions_relations = bidict({0:0, 1:2, 2:3})    # model:dataset
        self.default_y = 0
        self.output_size = len(actions)
        self.dim = dim[0]
        self.vid_inputs = dim[1]
        self.vid_dim = np.array(np.append(np.array(vid_dim).astype(int), sum(1 for e in self.vid_inputs if e is not 0))).astype(int)
        self.state_size = [self.dim, self.vid_dim]
        self.process_string = '\r Timestep: {timestep}/{dataset_len} - {percentage}% | total:{total_timesteps}/{total_len} - {total_percentage}% | model_actions avg: {model_actions_avg} | dataset time: {total_time} | ' \
                              'average step time: {step_time} | remaining time: dataset-{time_left} total-{total_time_left}'

        #model_subname += "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        q_name = "DQN_qnetwork" #+ "_" + model_subname
        t_name = "DQN_target" #+ "_" + model_subname

        self.datasetManager = dataset_manager.DatasetManager(dataset_type = dataset_type, subname=subname,
                                                             datasets_directory=datasets_directory, type=type) #C:/Users/ivana/OneDrive/Coding/ML/Com_Vis/car_project/Datasets

        if model_type=="DQN": self.Agent = Agents.DQN(state_size = self.state_size, actions=actions, optimizer=optimizer, models_directory="C:/ML_car/Models",
                                load_model=load_model, models_names=bidict({"q_name":q_name, "t_name":t_name}), model_subname=self.model_subname)
        elif model_type=="Actor_Critic": self.Agent = Agents.Actor_Critic(state_size = self.state_size, actions=actions, optimizer=optimizer,
                                models_directory="C:/ML_car/Models", load_model=load_model, model_name="Actor_critic")
        self.Agent.visualise_model()

    def simulate_on_dataset_2(self, file_name="", visualise=False):

        if file_name == "":dataset = self.datasetManager.dataset
        else:dataset = np.load(self.datasetManager.datasets_directory + '/' + file_name, allow_pickle=True)

        model_actions = []
        start = time.time()

        frame = dataset[0][0]
        distance = dataset[0][7]
        state = self.datasetManager.preprocessManager.state_preprocess(frame=frame, distance=distance)

        for timestep in range(1, len(dataset)-1):                                           # -- ***

            # Calculating action
            if self.model_type=="DQN": action = self.Agent.act(state)
            elif self.model_type=="Actor_Critic":
                action_probs, critic_value = self.Agent.act(state)
                action = np.random.choice(self.output_size, p=np.squeeze(action_probs))
            model_actions.append(action)
            # Take action
            # -                                                                             # -- ***

            # Receive next state                                                            # -- ***
            row = dataset[timestep]
            frame = row[0]
            distance = row[7]

            # Preprocess nextstate
            next_state = self.datasetManager.preprocessManager.state_preprocess(frame=frame, distance=distance)
            #print(state)

            # Detect objects and calculate reward
            objects=[]
            if self.type == "explorer": objects = self.datasetManager.preprocessManager.objects_detection(frame=frame)
            reward=0
            if ("last_frame" in locals()): reward = self.datasetManager.preprocessManager.reward_calculator(frame=frame, last_frame=last_frame,
                                                                                                            distance=distance, objects=objects)
            # Storing
            if self.model_type=="DQN": self.Agent.store(state, action, reward, next_state)
            elif self.model_type=="Actor_Critic": self.Agent.store(critic_value, action_probs, action, reward)

            state = next_state

            if self.model_type == "DQN" and len(self.Agent.expirience_replay) > self.batch_size: self.Agent.retrain(self.batch_size)
            elif self.model_type == "Actor_Critic" and len(self.Agent.rewards_history > self.batch_size): self.Agent.retrain()

            # Aligning target model if on second last state
            if (timestep == len(dataset) - 2 and self.model_type=="DQN"): self.Agent.alighn_target_model()
            if (timestep % 10 == 0): self.Agent.save_model(subname=self.model_subname)

            # Visualisation
            if visualise:
                sys.stdout.write(self.process_string.format(
                    timestep=timestep, dataset_len=len(dataset), percentage=round(timestep / len(dataset) * 100, 2),
                    model_actions_avg=round(np.average(model_actions), 2),
                    total_time=time.strftime('%H:%M:%S', time.gmtime(time.time() - start)),
                    step_time=round((time.time() - start) / timestep, 2), time_left=time.strftime('%H:%M:%S',time.gmtime((time.time() - start) / timestep * (len(dataset) - timestep))),
                    total_timesteps=self.total_timesteps, total_len=self.total_len,
                    total_percentage=round((self.total_timesteps) / self.total_len * 100, 2),
                    total_time_left=time.strftime('%H:%M:%S', time.gmtime(
                        (time.time() - start) / timestep * (self.total_len - self.total_timesteps)))))
                sys.stdout.flush()
                if (timestep % 50 == 0): print()

            last_frame = frame

    def simulate_on_dataset(self, file_name="", visualise=False):

        if file_name=="": dataset=self.datasetManager.dataset
        else: dataset = np.load(self.datasetManager.datasets_directory + '/' + file_name, allow_pickle=True)

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
            if(timestep==len(dataset)-2 and self.model_type=="DQN"): self.Agent.alighn_target_model()
            if(timestep%10==0): self.Agent.save_model(subname=self.model_subname)

            # Visualisation
            if visualise:
                sys.stdout.write(self.process_string.format(
                    timestep=timestep, dataset_len=len(dataset), percentage=round(timestep/len(dataset)*100, 2),
                    model_actions_avg=round(np.average(model_actions),2), total_time=time.strftime('%H:%M:%S', time.gmtime(time.time()-start)),
                    step_time=round((time.time()-start)/timestep,2), time_left=time.strftime('%H:%M:%S', time.gmtime((time.time()-start)/timestep*(len(dataset)-timestep))),
                    total_timesteps=self.total_timesteps, total_len=self.total_len, total_percentage=round((self.total_timesteps)/self.total_len*100, 2),
                    total_time_left=time.strftime('%H:%M:%S', time.gmtime((time.time()-start)/timestep*(self.total_len-self.total_timesteps)))))
                sys.stdout.flush()
                if(timestep%50==0): print()

            timestep+=1
            self.total_timesteps += 1

    def get_state_row_data(self, row, dim=[], vid_inputs=[]):

        if dim==[]: dim = self.dim
        if vid_inputs==[]: vid_inputs = self.vid_inputs

        num_data = []
        vid_data = []
        state = []
        if (dim[0] != 0): num_data.append(row[4][0]) #np.array #tf.convert_to_tensor(row[4][0], dtype=tf.int64) #np.asarray(row[4][0]).astype(np.float32)
        if (dim[1] != 0): num_data.append(row[5])
        if (dim[2] != 0): num_data.append(row[7])

        if (vid_inputs[0] != 0): vid_data.append(row[1][:,:,0])
        if (vid_inputs[1] != 0): vid_data.append(row[1][:,:,1])
        if (vid_inputs[2] != 0): vid_data.append(row[1][:,:,2])
        if (vid_inputs[3] != 0): vid_data.append(row[2])
        if (vid_inputs[4] != 0): vid_data.append(row[3])

        if vid_data!=[]:vid_data=np.array([vid_data])
        for num_element in num_data: state.append(np.array([num_element]))
        state.append(vid_data)

        if len(state)==0: state=state[0]

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
