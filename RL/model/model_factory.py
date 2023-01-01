from model.dqn.DQN import DQN
from model.attention.transformer import Transformer


def get_model(model_name):
    if model_name == "dqn":
        return DQN
    if model_name == "transformer":
        return Transformer
