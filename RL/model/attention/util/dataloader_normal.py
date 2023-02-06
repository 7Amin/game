import os
import pandas as pd
from torch.utils.data import Dataset


class NormalDataLoader(Dataset):
    def __init__(self, minibatch):
        self.minibatch = minibatch

    def __len__(self):
        return len(self.minibatch)

    def __getitem__(self, idx):
        state, action, reward, next_state, done = self.minibatch[idx]


        return (state, action, next_state, done), reward
