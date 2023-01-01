from collections import deque
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.optimizers import adam_v2
import random
import numpy as np


class DQN:
    def __init__(self, state_size, action_size):
        # define the state size
        self.state_size = state_size

        # define the action size
        self.action_size = action_size

        # define the replay buffer
        self.replay_buffer = deque(maxlen=5000)

        # define the discount factor
        self.gamma = 0.9

        # define the epsilon value
        self.epsilon = 0.8

        # define the update rate at which we want to update the target network
        self.update_rate = 1000

        # define the main network
        self.main_network = self.build_network()

        # define the target network
        self.target_network = self.build_network()

        # copy the weights of the main network to the target network
        self.target_network.set_weights(self.main_network.get_weights())

    def build_network(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=4, padding='same', input_shape=self.state_size))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (4, 4), strides=2, padding='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())

        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=adam_v2.Adam())

        return model

    # We learned that we train DQN by randomly sampling a minibatch of transitions from the
    # replay buffer. So, we define a function called store_transition which stores the transition information
    # into the replay buffer

    def store_transistion(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    # We learned that in DQN, to take care of exploration-exploitation trade off, we select action
    # using the epsilon-greedy policy. So, now we define the function called epsilon_greedy
    # for selecting action using the epsilon-greedy policy.

    def epsilon_greedy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.action_size)

        Q_values = self.main_network.predict(state)

        return np.argmax(Q_values[0])

    def train(self, batch_size):

        # sample a mini batch of transition from the replay buffer
        minibatch = random.sample(self.replay_buffer, batch_size)

        # compute the Q value using the target network
        for state, action, reward, next_state, done in minibatch:
            if not done:
                target_Q = (reward + self.gamma * np.amax(self.target_network.predict(next_state)))
            else:
                target_Q = reward

            # compute the Q value using the main network
            Q_values = self.main_network.predict(state)

            Q_values[0][action] = target_Q

            # train the main network
            self.main_network.fit(state, Q_values, epochs=1, verbose=0)

    # update the target network weights by copying from the main network
    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())


