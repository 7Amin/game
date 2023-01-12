import tensorflow as tf
import argparse
import gym
import json
import os
import numpy as np
from util.state_prepresentor import preprocess_state
from model.model_factory import get_model
from keras import backend as K
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print("this is error Amin")
        print(e)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-e", "--num_episodes", type=int, default=250000, metavar='>= 0', help="Number of Episodes")
parser.add_argument("-t", "--num_time_steps", type=int, default=200000, metavar='>= 0', help="Number of time steps")
parser.add_argument("-u", "--update_rate", type=int, default=3, metavar='>= 0', help="Update Rate")
parser.add_argument("-seq", "--sequence_state", type=int, default=1, metavar='>= 0', help="Sequence State")
parser.add_argument("-g", "--game_name", type=str, default='Breakout-v4',
                    choices=["Breakout-v4", "BeamRider-v4", "Enduro-v4", "Pong-v4", "Qbert-v4",
                             "Seaquest-v4", "SpaceInvaders-v4"], help="Choose from list")
parser.add_argument("-b", "--batch_size", type=int, default=8, metavar='>= 0', help="Batch size")
parser.add_argument("-m", "--model", type=str, default="AxialAttentionWithoutPosition",
                    choices=["dqn",
                             "transformer",
                             "AxialAttentionWithoutPosition",
                             "AxialAttentionPosition",
                             "AxialAttentionPositionGate"],
                    help="Number of model")
# parser.add_argument("-lr", "--learning_rate", type=float, default=0.01,
#                     metavar='>= 0', help="Learning rate")
args = parser.parse_args()

np.random.seed(42)
tf.compat.v1.random.set_random_seed(42)
GAME_NAME = args.game_name

env = gym.make(GAME_NAME, render_mode="human")
# env = gym.make(GAME_NAME, render_mode="rgb_array")
state_size = (80, 80, 1)
action_size = env.action_space.n




num_episodes = args.num_episodes
num_time_steps = args.num_time_steps
batch_size = args.batch_size
update_rate = args.update_rate
model = get_model(args.model)

file_path = f"./json_files/{GAME_NAME}_{args.model}.json"
model_path = './checkpoints/{}_{}.h5'.format(args.model, GAME_NAME)


dqn = model(args.model, state_size, action_size, update_rate, model_path, args.sequence_state)
if os.path.exists(model_path):
    dqn.load_model()

done = False
time_step = 0
dictionary = dict()

old_episode_number = 0

if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        dictionary = json.loads(f.read())

    for data in dictionary:
        if old_episode_number <= int(data):
            old_episode_number = int(data)
            time_step = dictionary[data]['time_step']

# for each episode
for i in range(old_episode_number + 1, num_episodes):

    # set return to 0
    Return = 0

    # preprocess the game screen
    state = preprocess_state(env.reset()[0])

    # for each step in the episode
    for t in range(num_time_steps):

        # render the environment
        env.render()

        # update the time step
        time_step += 1

        # update the target network
        if time_step % dqn.update_rate == 0:
            print("frame number is {}".format(time_step))
            dqn.update_target_network()
            dictionary[i] = {
                'reward': Return,
                'time_step': time_step
            }
            with open(file_path, "w") as outfile:
                json.dump(dictionary, outfile)

        # select the action
        action = dqn.epsilon_greedy(state, time_step)

        # perform the selected action
        next_state, reward, done, truncated, _ = env.step(action)
        # preprocess the next state
        next_state = preprocess_state(next_state)

        # store the transition information
        dqn.store_transition(state, action, reward, next_state, done)

        # update current state to next state
        state = next_state

        # update the return
        Return += reward

        # if the episode is done then print the return
        if done:
            dictionary[i] = {
                'reward': Return,
                'time_step': time_step
            }
            print('Episode: ', i, ',' 'Return', Return)
            with open(file_path, "w") as outfile:
                json.dump(dictionary, outfile)
            break

        if len(dqn.replay_buffer) > batch_size * args.sequence_state:
            dqn.train(batch_size)
    if time_step > 250 * 1000 * 1000 + 10:
        break

print("frame number is {}".format(time_step))
