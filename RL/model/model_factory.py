from model.dqn.DQN import DQN


def get_model(model_name):
    if model_name == "dqn":
        return DQN
