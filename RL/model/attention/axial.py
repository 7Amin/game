import pdb
import math
import torch
import torch.nn as nn
from torch import optim
import cv2
import torch.nn.functional as F
import pdb
import matplotlib.pyplot as plt
import numpy as np
from model.attention.util.axial_attention import AxialAttentionModel, \
    AxialWithoutPositionBlock, AxialPositionGateBlock, AxialPositionBlock

import random
from model.basic_model import BasicModel


class BasicAxialModel(BasicModel):
    def __init__(self, name, state_size, action_size, update_rate, model_path, sequence_state=1):
        super().__init__(name, state_size, action_size, update_rate, model_path, sequence_state)
        self.name = name

        self.main_network = self.build_network()
        self.target_network = self.build_network()
        self.save_model_main()
        self.load_model_target()
        self.criterion = self._select_criterion()
        self.optimizer = self._select_optimizer()
        # self.target_network.set_weights(self.main_network.get_weights())

    def train(self, batch_size):
        super().train(batch_size=batch_size)
        self.clear_buffer()

    @staticmethod
    def _create_tensor(x):
        down_width = 128
        down_height = 128
        down_points = (down_width, down_height)
        x = cv2.resize(x[0], down_points, interpolation=cv2.INTER_LINEAR)
        x = np.expand_dims(x, axis=-1)
        x = np.expand_dims(x, axis=0)
        state_tensor = torch.tensor(x).float()
        state_tensor = state_tensor.permute(0, 3, 1, 2)
        return state_tensor

    def epsilon_greedy(self, state, time_step):
        epsilon = self._get_epsilon(time_step)
        if random.uniform(0, 1) < epsilon:
            return np.random.randint(self.action_size)

        state_tensor = self._create_tensor(state)
        Q_values = self.main_network(state_tensor)
        Q_values = torch.reshape(Q_values, (1, Q_values.shape[1]))
        return torch.argmax(Q_values[0]).item()

    def update_target_network(self):
        # self.target_network.load_state_dict(self.main_network.state_dict())
        self.save_model_main()
        self.load_model_target()
        # self.target_network.set_weights(self.main_network.get_weights())
        # self.target_network.save(self.model_path)
        print("target_network is updated")

    def save_model_target(self):
        torch.save(self.target_network, self.model_path)
        # self.target_network.save(self.model_path)

    def save_model_main(self):
        torch.save(self.main_network, self.model_path)
        # self.main_network.save(self.model_path)

    def load_model_target(self):
        self.target_network = torch.load(self.model_path)
        self.target_network.eval()
        # self.target_network.load_weights(self.model_path)

    def load_model_main(self):
        self.main_network = torch.load(self.model_path)
        self.main_network.eval()
        # self.main_network.load_weights(self.model_path)

    def _select_optimizer(self):
        model_optim = optim.Adam(self.main_network.parameters(), lr=0.001)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, batch_size):
        minibatch = random.sample(self.replay_buffer, batch_size)

        # compute the Q value using the target network
        for state, action, reward, next_state, done in minibatch:
            if not done:
                Q_values = self.target_network(self._create_tensor(next_state))
                Q_values = torch.reshape(Q_values, (1, Q_values.shape[1]))
                v = torch.max(Q_values[0]).item()
                target_Q = (reward + self.gamma * v)
            else:
                target_Q = reward

            # compute the Q value using the main network

            state_tensor = self._create_tensor(state)
            Q_values = self.main_network(state_tensor)
            Q_values = torch.reshape(Q_values, (1, Q_values.shape[1]))
            Q_hat_values = Q_values
            Q_values[0][action] = target_Q
            # csv_logger = CSVLogger('./log/log.csv', append=True, separator=';')
            # train the main network
            # self.main_network.fit(state, Q_values, epochs=2, verbose=0)  # callbacks=[csv_logger]
            loss = self.criterion(Q_hat_values, Q_values)
            loss.backward()
            self.optimizer.zero_grad()
            self.optimizer.step()


class AxialAttentionWithoutPositionModel(BasicAxialModel):
    def build_network(self):
        model = AxialAttentionModel(AxialWithoutPositionBlock, [1, 2, 4, 1],
                                    num_classes=self.action_size,
                                    s=0.125)
        return model


class AxialAttentionPositionModel(BasicAxialModel):
    def build_network(self):
        model = AxialAttentionModel(AxialPositionBlock, [1, 2, 4, 1],
                                    num_classes=self.action_size,
                                    s=0.125)
        return model


class AxialAttentionPositionGateModel(BasicAxialModel):
    def build_network(self):
        model = AxialAttentionModel(AxialPositionGateBlock, [1, 2, 4, 1],
                                    num_classes=self.action_size,
                                    s=0.125)
        return model
