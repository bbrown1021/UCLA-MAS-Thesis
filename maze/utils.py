from dataclasses import dataclass, field
import gym
import imageio
from IPython.display import Image
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import rl.callbacks
from rl.policy import Policy
import tensorflow as tf

class EpisodeLogger(rl.callbacks.Callback):
    r""" Callback of DQN test results. """
    def __init__(self):
        self.observations = {}
        self.rewards = {}
        self.actions = {}

    def on_episode_begin(self, episode, logs):
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        
    def on_step_end(self, step, logs):
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])

@dataclass
class Object:
  r""" Defines the data properties of objects (obstacles, empty space, agents, and goals)."""
  name: str
  value: int
  rgb: tuple
  impassable: bool
  positions: list = field(default_factory=list)

@dataclass
class RGBColor:
    r""" Defines the color schema for each object. """
    obstacle = (160, 160, 160)
    free = (224, 224, 224)
    agent = (51, 153, 255)
    goal = (51, 255, 51)

def build_model(env, nb_actions):
    r"""Builds a basic neural network for DQNAgent to approximates the state-value function. """
    model = Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(tf.keras.layers.Dense(16))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(nb_actions))
    model.add(tf.keras.layers.Activation('linear'))
    return model

def dqn_results(cb_ep, bc):
    r""" Extracts DQN callback to evaluate if agent converges to a single path. """
    dqn_actions = cb_ep.actions 
    steps = len(list(dqn_actions.values())[0])

    res = True 
    initial_value = list(dqn_actions.values())[0]
    for i in dqn_actions:
        if dqn_actions[i] != initial_value:
            res = False 
            break
    if res == True:
        if bc == True:
            print(f"DQN w/ Expert Agent Successfully Trained, solves maze in {steps} actions!!!")
        else:
            print(f"DQN Successfully Trained, solves maze in {steps} actions!!!")
    else:
        print("DQN did not converge.")

def extract_optimal_info(sample_maze):
    r"""Retrieves optimal dijkstra actions and determines the proportion of each action in solved environment. """
    file_name = "./dijkstra/dijkstra_results/maze%d/optimal_actions.npy" % sample_maze
    optimal_actions = list(np.load(file_name))
    res = (len([ele for ele in optimal_actions if ele > 1]) / len(optimal_actions))
    prob = [1-res, res]
    return prob

def save_video(env, filePath, actions, env_id):
    r""" Performs given actions in gym environment to retrieve total reward and save a video of the solved maze."""
    env = gym.wrappers.Monitor(env, filePath, force=True)
    env.reset()
    rewards = 0.0
    for action in actions:
        _, reward, _, _ = env.step(action)
        rewards += reward
    env.close()

    # saves a video (mp4) and gif of solved maze
    f = list(Path(filePath).glob('*.mp4'))[0]
    reader = imageio.get_reader(f)
    f = f'{filePath}{env_id}.gif'
    with imageio.get_writer(f, fps=1) as writer:
        [writer.append_data(img) for img in reader]
    Image(f)

def SB_test(env, model, episodes):
    r""" Performs optimal actions of Stable Baselines3 agents in gym environment."""
    for episode in range(1, episodes+1):
        obs = env.reset() # inital set of observations
        done = False
        score = 0 #running score counter
        steps = 0 
        
        while not done:
            env.render() # view the graphical environment
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action) # unpack the environmental results of chosen action
            score+=reward # sum reward
            steps += 1
        print(f'Episode:{episode} Score:{score}')
    env.close()
    return steps

def show_image(env):
    img = env.render('rgb_array')
    plt.imshow(img)
    plt.show()
    plt.close()
    env.close()
