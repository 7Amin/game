from collections import deque
import random
import numpy as np


class BasicModel:
    def __init__(self, state_size, action_size, update_rate):
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

        # define the main network
        self.main_network = self.build_network()

        # define the target network
        self.target_network = self.build_network()

        # copy the weights of the main network to the target network
        self.target_network.set_weights(self.main_network.get_weights())

    def build_network(self):
        raise NotImplemented

    def store_transition(self, state, action, reward, next_state, done):
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

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())
