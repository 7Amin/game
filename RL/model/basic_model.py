from collections import deque
import random
import math
import numpy as np
from keras.callbacks import CSVLogger


class BasicModel:
    def __init__(self, state_size, action_size, update_rate):
        self.name = "Base"
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
        self.update_rate = update_rate

    def build_network(self):
        raise NotImplemented

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

        # We learned that in DQN, to take care of exploration-exploitation trade off, we select action
        # using the epsilon-greedy policy. So, now we define the function called epsilon_greedy
        # for selecting action using the epsilon-greedy policy.

    def epsilon_greedy(self, state, time_step):
        epsilon = self.epsilon - math.floor(time_step / 5000) / 200 * 300
        if random.uniform(0, 1) < epsilon:
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
            # csv_logger = CSVLogger('./log/log.csv', append=True, separator=';')
            # train the main network
            self.main_network.fit(state, Q_values, epochs=3, verbose=0)  # callbacks=[csv_logger]

    def update_target_network(self, time_step, GAME_NAME):
        self.target_network.set_weights(self.main_network.get_weights())
        self.target_network.save('./checkpoints/{}_{}_{}.h5'.format(self.name, GAME_NAME, time_step))
