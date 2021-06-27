
import numpy as np
import random
from collections import deque
from bidict import bidict
import os

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import keras

import ML_training.Model as Models



class DQN():

    def __init__(
            self,
            state_size, actions, optimizer, models_directory, load_model, models_names=bidict({"q_name":"DQN_qnetwork", "t_name":"DQN_target"})):
        """Initialize."""
        super().__init__()

        # Initialize atributes
        self._state_size = state_size
        self._action_size = len(actions)
        self.actions = actions
        self._optimizer = optimizer
        self.models_directory = models_directory
        self.models_names = models_names

        self.expirience_replay = deque(maxlen=2000)

        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon = 0.1
        model_string = "{directory}/{name}"
        model_full_path=bidict({"q_path":model_string.format(directory=models_directory, name=models_names["q_name"]),
                                "t_path":model_string.format(directory=models_directory, name=models_names["t_name"])})

        #print(str(state_size[0][1]) + " | " + str(state_size[1][2])  + " | " + str(self._action_size))
        # Build networks
        if(load_model and os.path.exists(model_full_path["q_path"]) and os.path.exists(model_full_path["t_path"])):
            self.q_network = keras.models.load_model(model_full_path["q_path"])
            self.target_network = keras.models.load_model(model_full_path["t_path"])
        else:
            self.Model_manager = Models.DNN(c_dim = state_size[0][0], a_dim = state_size[0][1], d_dim = state_size[0][2],
                                width = state_size[1][0], height = state_size[1][1], depth = state_size[1][2], output_size = self._action_size)

            self.q_network = self.Model_manager.model
            self.target_network = self.Model_manager.model
        self.alighn_target_model()

    def store(self, state, action, reward, next_state):
        self.expirience_replay.append((state, action, reward, next_state))

    def _build_compile_model(self):
        model = Sequential()
        model.add(Embedding(self._state_size, 10, input_length=1))
        model.add(Reshape((10,)))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))

        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.actions)

        q_values = self.q_network.predict(state)
        return np.argmax(q_values[0])

    def retrain(self, batch_size):
        minibatch = random.sample(self.expirience_replay, batch_size)

        for state, action, reward, next_state in minibatch:

            target = self.q_network.predict(state)

            #if terminated:
            #    target[0][action] = reward
            #else:
            t = self.target_network.predict(next_state)
            target[0][action] = reward + self.gamma * np.amax(t)

            self.q_network.fit(state, target, epochs=1, verbose=0)

    def save_model(self, subname=""):

        tf.keras.models.save_model(
            self.q_network, "{directory}/Training/{subname}/{name}".format(directory=self.models_directory,name=self.models_names["q_name"],
                                                                                subname=subname), overwrite=True, include_optimizer=True,
                                                                 save_format=None, signatures=None, options=None, save_traces=True)
        tf.keras.models.save_model(
            self.target_network, "{directory}/Training/{subname}/{name}".format(directory=self.models_directory,name=self.models_names["t_name"],
                                                                                subname=subname), overwrite=True, include_optimizer=True,
                save_format=None,signatures=None, options=None, save_traces=True)

    def visualise_model(self):
        self.q_network.summary()
        self.target_network.summary()
        plot_model(model=self.q_network, to_file="{directory}/img/{name}.png".format(directory=self.models_directory, name=self.models_names["q_name"]), show_shapes=True)
        plot_model(model=self.target_network, to_file="{directory}/img/{name}.png".format(directory=self.models_directory, name=self.models_names["t_name"]), show_shapes=True)


class DNN():

  def __init__(
          self,
          dim, vid_dim, output_size):
    """Initialize."""
    super().__init__()

    #self.c_dim, self.a_dim, self.d_dim = dim
    self.Model = Models.DNN(c_dim = dim[0], a_dim = dim[1], d_dim = dim[2],
                            width = vid_dim[0], height = vid_dim[1], depth = vid_dim[2], output_size = output_size)

  #def run(self):
