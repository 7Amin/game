import tensorflow as tf
import argparse
import cv2
import random
import gym
import numpy as np
from util.state_prepresentor import preprocess_state
from model.model_factory import get_model


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-e", "--num_episodes", type=int, default=2500, metavar='>= 0', help="Number of Episodes")
parser.add_argument("-t", "--num_time_steps", type=int, default=20000, metavar='>= 0', help="Number of time steps")
parser.add_argument("-u", "--update_rate", type=int, default=500, metavar='>= 0', help="Update Rate")
parser.add_argument("-g", "--game_name", type=str, default='Breakout-v4',
                    choices=["Breakout-v4", "BeamRider-v4", "Enduro-v4", "Pong-v4", "Qbert-v4",
                             "Seaquest-v4", "SpaceInvaders-v4"], help="Choose from list")
parser.add_argument("-b", "--batch_size", type=int, default=8, metavar='>= 0', help="Batch size")
parser.add_argument("-m", "--model", type=str, default="dqn", choices=["dqn"],
                    help="Number of model")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.01,
                    metavar='>= 0', help="Learning rate")
args = parser.parse_args()

np.random.seed(42)
tf.compat.v1.random.set_random_seed(42)
GAME_NAME = args.game_name

env = gym.make(GAME_NAME, render_mode="human")
state_size = (88, 80, 1)
action_size = env.action_space.n


num_episodes = args.num_episodes
num_time_steps = args.num_time_steps
batch_size = args.batch_size
update_rate = args.update_rate
model = get_model(args.model)

dqn = model(state_size, action_size, update_rate)
done = False
time_step = 0

# for each episode
for i in range(num_episodes):

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
            dqn.update_target_network()

        # select the action
        action = dqn.epsilon_greedy(state)

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
            print('Episode: ', i, ',' 'Return', Return)
            break

        #if the number of transistions in the replay buffer is greater than batch size
        #then train the network
        if len(dqn.replay_buffer) > batch_size:
            dqn.train(batch_size)

