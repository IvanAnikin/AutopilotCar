
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
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

import ML_training.Model as Models


class Actor_Critic():

    def __init__(
            self,
            state_size, actions, optimizer, models_directory, load_model, model_name="Actor_Critic",
            model_subname=""):
        """Initialize."""
        super().__init__()

        # Initialize atributes
        self._state_size = state_size
        self._action_size = len(actions)
        self.actions = actions
        self._optimizer = optimizer
        self.models_directory = models_directory
        self.model_name = model_name
        self.model_subname = model_subname

        self.expirience_replay = deque(maxlen=2000)
        self.action_probs_history = []
        self.critic_value_history = []
        self.rewards_history = []
        self.huber_loss = Huber()
        self.optimizer = Adam(learning_rate=0.01)
        self.gamma = 0.99  # Discount factor for past rewards
        self.eps = np.finfo(np.float32).eps.item()

        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon = 0.1
        dir = "{directory}/Trained/673243b/All/"
        if model_subname != "": dir += "{subname}/"
        dir+="{name}.h5"
        model_full_path=dir.format(directory=models_directory, name=model_name, subname=model_subname)

        #print(str(state_size[0][1]) + " | " + str(state_size[1][2])  + " | " + str(self._action_size))
        # Build networks
        self.Model_manager = Models.DNN(c_dim=state_size[0][0], a_dim=state_size[0][1], d_dim=state_size[0][2],
                                        width=state_size[1][0], height=state_size[1][1], depth=state_size[1][2],
                                        output_size=self._action_size, critic=True)

        self.model = self.Model_manager.model
        print("Models created")

        if load_model and os.path.exists(model_full_path):
            self.model.load_weights(model_full_path)
            print("Weights loaded | {path}".format(path=model_full_path))

    def act(self, state):

        return self.model(state)

    def store(self, critic_value, action_probs, action, reward):
        self.critic_value_history.append(critic_value[0, 0])
        self.action_probs_history.append(tf.math.log(action_probs[0, action]))
        self.rewards_history.append(reward)

    def save_model(self, subname=""):
        dir="{directory}/Training/Actor_Critic/"
        if subname!="": dir+="{subname}/"
        name = dir.format(directory=self.models_directory, subname=subname)
        if not os.path.exists(name): os.mkdir(name)
        name+="{name}.h5"
        #name=name.format(name=self.models_names["q_name"])
        self.model.save_weights(name.format(name=self.model_name))

    def visualise_model(self):
        self.model.summary()
        plot_model(model=self.model, to_file="{directory}/img/Actor_Critic/{name}_{subname}.png".format(directory=self.models_directory, name=self.model_name, subname=self.model_subname), show_shapes=True)

    def retrain(self):
        with tf.GradientTape() as tape:
            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            returns = []
            discounted_sum = 0
            for r in self.rewards_history[::-1]:
                discounted_sum = r + self.gamma * discounted_sum
                returns.insert(0, discounted_sum)

            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)
            returns = returns.tolist()

            # Calculating loss values to update our network
            history = zip(self.action_probs_history, self.critic_value_history, returns)
            actor_losses = []
            critic_losses = []
            for log_prob, value, ret in history:
                # At this point in history, the critic estimated that we would get a
                # total reward = `value` in the future. We took an action with log probability
                # of `log_prob` and ended up recieving a total reward = `ret`.
                # The actor must be updated so that it predicts an action that leads to
                # high rewards (compared to critic's estimate) with high probability.
                diff = ret - value
                actor_losses.append(-log_prob * diff)  # actor loss

                # The critic must be updated so that it predicts a better estimate of
                # the future rewards.
                critic_losses.append(
                    self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )

            # Backpropagation
            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            # Clear the loss and reward history
            self.action_probs_history.clear()
            self.critic_value_history.clear()
            self.rewards_history.clear()


class DQN():

    def __init__(
            self,
            state_size, actions, optimizer, models_directory, load_model, models_names=bidict({"q_name":"DQN_qnetwork", "t_name":"DQN_target"}), model_subname=""):
        """Initialize."""
        super().__init__()

        # Initialize atributes
        self._state_size = state_size
        self._action_size = len(actions)
        self.actions = actions
        self._optimizer = optimizer
        self.models_directory = models_directory
        self.models_names = models_names
        self.model_subname = model_subname

        self.expirience_replay = deque(maxlen=2000)

        # Initialize discount and exploration rate
        self.gamma = 0.6
        self.epsilon = 0.1
        dir = "{directory}/Trained/673243b/All/"
        if model_subname != "": dir += "{subname}/"
        dir+="{name}.h5"
        model_full_path=bidict({"q_path":dir.format(directory=models_directory, name=models_names["q_name"], subname=model_subname),
                                "t_path":dir.format(directory=models_directory, name=models_names["t_name"], subname=model_subname)})

        #print(str(state_size[0][1]) + " | " + str(state_size[1][2])  + " | " + str(self._action_size))
        # Build networks
        self.Model_manager = Models.DNN(c_dim=state_size[0][0], a_dim=state_size[0][1], d_dim=state_size[0][2],
                                        width=state_size[1][0], height=state_size[1][1], depth=state_size[1][2],
                                        output_size=self._action_size)
        self.q_network = self.Model_manager.model
        self.target_network = self.Model_manager.model
        print("Models created")

        if(load_model and os.path.exists(model_full_path["q_path"]) and os.path.exists(model_full_path["t_path"])):
            #self.q_network = keras.models.load_model(model_full_path["q_path"])
            #self.target_network = keras.models.load_model(model_full_path["t_path"])
            self.q_network.load_weights(model_full_path['q_path'])
            self.target_network.load_weights(model_full_path['t_path'])
            print("Weights loaded | {q_path}".format(q_path=model_full_path["q_path"]))
            print("Weights loaded | {t_path}".format(t_path=model_full_path["t_path"]))

        self.alighn_target_model()

    def store(self, state, action, reward, next_state):
        self.expirience_replay.append((state, action, reward, next_state))

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
        dir="{directory}/Training/"
        if subname!="": dir+="{subname}/"
        name = dir.format(directory=self.models_directory, subname=subname)
        if not os.path.exists(name): os.mkdir(name)
        name+="{name}.h5"
        #name=name.format(name=self.models_names["q_name"])
        self.q_network.save_weights(name.format(name=self.models_names["q_name"]))
        self.target_network.save_weights(name.format(name=self.models_names["t_name"]))
        '''
        tf.keras.models.save_model(
            self.q_network, dir.format(directory=self.models_directory,name=self.models_names["q_name"],
                                                                                subname=subname), overwrite=True, include_optimizer=True,
                                                                 save_format=None, signatures=None, options=None, save_traces=True)
        tf.keras.models.save_model(
            self.target_network, dir.format(directory=self.models_directory,name=self.models_names["t_name"],
                                                                                subname=subname), overwrite=True, include_optimizer=True,
                save_format=None,signatures=None, options=None, save_traces=True)
        '''
    def visualise_model(self):
        self.q_network.summary()
        #self.target_network.summary()
        plot_model(model=self.q_network, to_file="{directory}/img/{name}_{subname}.png".format(directory=self.models_directory, name=self.models_names["q_name"], subname=self.model_subname), show_shapes=True)
        #plot_model(model=self.target_network, to_file="{directory}/img/{name}.png".format(directory=self.models_directory, name=self.models_names["t_name"]), show_shapes=True)


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
