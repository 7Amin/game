from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Conv2D
from tensorflow.python.keras.optimizers import adam_v2

from model.basic_model import BasicModel
import random
import numpy as np


class DQN(BasicModel):
    def __init__(self, state_size, action_size, update_rate):
        super().__init__(state_size, action_size, update_rate)
        self.name = "DQN"
        self.main_network = self.build_network()
        self.target_network = self.build_network()
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

        model.compile(loss='mse', optimizer=adam_v2.Adam(learning_rate=0.001))

        return model

