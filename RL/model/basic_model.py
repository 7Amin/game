from collections import deque


class BasicModel(object):
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
        raise NotImplemented

    def train(self, batch_size):
        raise NotImplemented

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())
