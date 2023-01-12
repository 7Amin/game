from model.dqn.DQN import DQN
from model.attention.transformer import Transformer
from model.attention.axial import AxialAttentionWithoutPositionModel,\
    AxialAttentionPositionModel, AxialAttentionPositionGateModel


def get_model(model_name):
    if model_name == "dqn":
        return DQN
    if model_name == "transformer":
        return Transformer
    if model_name == "AxialAttentionWithoutPosition":
        return AxialAttentionWithoutPositionModel
    if model_name == "AxialAttentionPosition":
        return AxialAttentionPositionModel
    if model_name == "AxialAttentionPositionGate":
        return AxialAttentionPositionGateModel
