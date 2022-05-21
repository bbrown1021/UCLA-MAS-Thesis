""" 
Trains multiple RL agents on a customizable maze environment.

- Default Maze:
    - 'maze/maze_generator.py' generates 20 x 20 mazes with random obstacles
    - select a pre-generated maze from 'maze/sample_envs'
- Default Environment:
    - start location: top left
    - end location: bottom right
    - maximum episode steps: 1000
    - reward system:
        - minus 0.1 for selecting an invalid state (obstacle)
        - minus 0.001 for selecting a valid state that is NOT the goal
        - plus 10 for reaching the goal

- Pathfinding Methods:
    - Dijkstra's Algorithm
        - solves each maze in 34 steps
        - achieves 9.967 reward

- Reinforcement Learning Methods:
    - A2C
    - PPO
    - DQN
    - DQN w/ Expert (Behavioral Cloning)
"""

############################
# Import Necessary Modules #
############################

import numpy as np
import os

# OpenAI Gym
import gym
from gym.spaces import Box, Discrete

# Custom Environment
from maze import BaseMaze, BaseEnv, VonNeumannMotion, EpisodeLogger, Object, RGBColor as color
from maze import build_model, dqn_results, extract_optimal_info, save_video, SB_test, show_image
from dijkstra.dijkstra_solver import dijkstra_solver

# A2C / PPO
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# DQN
from keras.optimizers import adam_v2
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import Policy, EpsGreedyQPolicy, GreedyQPolicy
import tensorflow as tf

# ignore internal tensorflow messages
from multiprocessing import freeze_support
import warnings
warnings.filterwarnings("ignore")

###############################
# Generate Custom Environment #
###############################

# User selects sample maze (0,1,2,3) from 'maze/sample_envs'
while True:
  try:
    options = [0, 1, 2, 3]
    sample_maze = int(input("Select a sample maze (0, 1, 2, or 3): "))
    if sample_maze in options:
      break;
    else:
      print("Invalid Selection")      
  except ValueError:
    print("Invalid Selection")
    continue

# Define a starting location and end goal
start_idx = [[1, 1]] # top left corner
goal_idx = [[18, 18]] # bottom left corner

# Load pre-generated random maze 
x = np.load('maze/sample_envs/maze%d.npy' % sample_maze)
env_id = 'RandomShapeMaze-v%d' % sample_maze

# Creates maze with four objects
class Maze(BaseMaze):
    @property
    def size(self):
        r"""Returns the dimensions of maze (height, width). """
        return x.shape
    
    def make_objects(self):
        r"""Define and return the list of objects. """
        free = Object('free', 0, color.free, False, np.stack(np.where(x == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(x == 1), axis=1))
        agent = Object('agent', 2, color.agent, False, [])
        goal = Object('goal', 3, color.goal, False, [])
        return free, obstacle, agent, goal
        
# Creates environment with predefined random maze
class Env(BaseEnv):
    def __init__(self):
        super().__init__() # use the functionality of defined render modes
        
        # custom maze and observation space for environment
        self.maze = Maze()
        self.observation_space = Box(low=0, high=len(self.maze.objects) , shape=self.maze.size, dtype=np.uint8)
        
        # custom motion and action space for agent
        self.motions = VonNeumannMotion()
        self.action_space = Discrete(len(self.motions))
        
    def step(self, action):
        r"""environment updates given the agent's action"""

        # (N,E,S,W) direction chosen from action
        motion = self.motions[action] 

        # extract current position of agent
        current_position = self.maze.objects.agent.positions[0] 

        # update position of agent based on new action
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]] 

        
        # checks that new position is valid
        # if FALSE, agent position does NOT update but reward STILL UPDATES
        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]
        # reward system
        if self._is_goal(new_position): # agent reached goal
            reward = +10 # huge reward
            done = True
        elif not valid: # agent chose an invalid position
            reward = -0.1 # punishment
            done = False
        else: # agent chose a valid position but did not find the goal yet
            reward = -0.001 # small punishment
            done = False
        return self.maze.to_value(), reward, done, {}
        
    def reset(self):
        r"""sets the new agent and goal positions"""
        self.current_step = 0
        self.maze.objects.agent.positions = start_idx
        self.maze.objects.goal.positions = goal_idx
        return self.maze.to_value()
    
    def _is_valid(self, position):
        r"""boolean checks for valid agent positions"""

        # verify agent is still within the observation space (x and y coordinates)
        within_inner_edge = position[0] >= 0 and position[1] >= 0 # nonegative
        within_outer_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1] # maze size

        # verifies agent's current position is passable (not entering an obstacle position)
        passable = not self.maze.to_impassable()[position[0]][position[1]]

        return within_inner_edge and within_outer_edge and passable
    
    def _is_goal(self, position):
        out = False # default is FALSE for not reaching goal

        # checks if agent is in the same position as the goal
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break # if so, for loop check breaks
        return out 
    
    def get_image(self):
        r"""Returns the rgb color tuple for entire environment"""
        return self.maze.to_rgb()

# Creates custom environment with OpenAI gym module
gym.envs.register(id=env_id, entry_point=Env, max_episode_steps=1000)
env = gym.make(env_id)
env.reset()
print(f"------ Successfully Generated Maze: {env_id} ------")
# show_image(env)

print("--------------------------------------------")
print("------ Dijkstra Pathfinding Algorithm ------")
print("--------------------------------------------")

# extract maze information for dijkstra solver in 'agent'
impassable_array = env.unwrapped.maze.to_impassable()
motions = env.unwrapped.motions
start = env.maze.objects.agent.positions[0]
goal = env.unwrapped.maze.objects.goal.positions[0]

# solve maze with dijkstra's algorithm
actions = dijkstra_solver(impassable_array, motions, start, goal)

# save optimal actions
file_name = "./dijkstra/dijkstra_results/maze%d/optimal_actions.npy" % sample_maze
np.save(file_name, actions)

# view actions
print(f'Optimal Actions: {actions}')
print(f'Total Actions: {len(actions)}')

# save visuals
save_video(env,'./dijkstra/dijkstra_results/maze%d/' % sample_maze , actions, env_id)

#####################################
# Training and Evaluating RL Agents #
#####################################

# extract environment details for training
nb_actions = env.action_space.n

# total training steps used in models found in 'training/saved_models'
# note: not all agents were capable of solving a maze
if sample_maze == 0:
    a2c_steps = 50000
    ppo_steps = 35000
    dqn_steps = 30000
    bc_dqn_learn = 1000
elif sample_maze == 1:
    a2c_steps = 60000
    ppo_steps = 40000
    dqn_steps = 30000
    bc_dqn_learn = 2000
elif sample_maze == 2:
    a2c_steps = 70000
    ppo_steps = 80000
    dqn_steps = 40000
    bc_dqn_learn = 3000
else:
    a2c_steps = 80000
    ppo_steps = 120000
    dqn_steps = 35000
    bc_dqn_learn = 4000

print("----------------------")
print("-------- A2C ---------")
print("----------------------")

# # train
# print(f"Training A2C for {a2c_steps} training steps...")
# env = DummyVecEnv([lambda: env]) # wrap in Dummy Vectorized environment
# model = A2C("MlpPolicy", env, gamma = 0.995, learning_rate = 0.0001, ent_coef = 0.01, verbose = 1)
# model.learn(total_timesteps=a2c_steps)

# # save
# A2C_Path = os.path.join('training', 'saved_models', 'A2C', 'A2C_MLP_model_v%d' % sample_maze)
# print(f"Saving A2C in {A2C_Path}...")
# model.save(A2C_Path)

# load 
A2C_Path = os.path.join('training', 'saved_models', 'A2C', 'A2C_MLP_model_v%d' % sample_maze)
model = A2C.load(A2C_Path, env=env)
print(f'Loading A2C {model} after training for {a2c_steps} steps...')

# evaluate and test trained model
results = evaluate_policy(model, env, n_eval_episodes=5, render=False)
print(f'Evaluation: {results}')
A2C_res = SB_test(env, model, 5)
if results[0] == 9.967:
    print(f"A2C Successfully Trained, solves maze in {A2C_res} actions!!!")
else:
    print("A2C did not converge.")

print("----------------------")
print("-------- PPO ---------")
print("----------------------")

# # train
# print(f"Training PPO for {ppo_steps} training steps...")
# env = DummyVecEnv([lambda: env]) # wrap in Dummy Vectorized environment
# model = PPO('MlpPolicy', env, learning_rate=0.005, gamma = 0.9, ent_coef = 0.01, verbose = 1)
# model.learn(total_timesteps=ppo_steps)

# # save
# PPO_Path = os.path.join('training', 'saved_models', 'PPO', 'PPO_MLP_model_v%d' % sample_maze)
# print(f"Saving PPO in {PPO_Path}...")
# model.save(PPO_Path)

# load 
PPO_Path = os.path.join('training', 'saved_models', 'PPO', 'PPO_MLP_model_v%d' % sample_maze)
model = PPO.load(PPO_Path, env=env)
print(f'Loading PPO {model} after training for {ppo_steps} steps...')

# evaluate
results = evaluate_policy(model, env, n_eval_episodes=5, render=False)
print(f'Evaluation: {results}')
PPO_res = SB_test(env, model, 5)
if results[0] == 9.967:
    print(f"PPO Successfully Trained, solves maze in {PPO_res} actions!!!")
else:
    print("PPO did not converge.")


print("----------------------")
print("-------- DQN ---------")
print("----------------------")

# initialize
lr = 1e-3 # learning rate
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
model = build_model(env, nb_actions) # neural network
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000, target_model_update=1e-2, policy=policy)
dqn.compile(adam_v2.Adam(learning_rate=lr), metrics=['mae'])

# # train 
# print(f"Training DQN for {dqn_steps} training steps...")
# dqn.fit(env, nb_steps=dqn_steps, verbose=2)

# # save
# print(f"Saving DQN...")
# dqn.save_weights('training/saved_models/DQN/maze%d/dqn_weights.h5f' % sample_maze, overwrite=True)

# load 
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100, target_model_update=1e-2, policy=policy)
dqn.compile(adam_v2.Adam(learning_rate=lr), metrics=['mae'])
dqn.load_weights('training/saved_models/DQN/maze%d/dqn_weights.h5f' % sample_maze)
print(f'Loading DQN {dqn} after training for {dqn_steps} steps...')

# evaluate
cb_ep = EpisodeLogger()
dqn.test(env, nb_episodes=5, visualize=True, callbacks=[cb_ep])
dqn_results(cb_ep, False)

print("---------------------------------------")
print("------ DQN w/ Behavioral Cloning ------")
print("---------------------------------------")

# initialize
lr = 1e-3 # learning rate
memory = SequentialMemory(limit=50000, window_length=1)
model = build_model(env, nb_actions) # neural network

# # expert 
# class ExpertPolicy(Policy):
#     """ Custom policy implements action selection:
#     - 95% random selection of action 1 or 3 dependent on prevalence of optimal actions in solved Dijkstra results
#     - 5% selects best action from current Q network
#     """
#     def select_action(self, q_values):
#         assert q_values.ndim == 1
#         nb_actions = q_values.shape[0]
#         prob = extract_optimal_info(sample_maze)

#         if np.random.uniform() < 0.95:
#             action = np.random.choice((1,3),p=prob)
#         else:
#             action = np.argmax(q_values)
#         return action
# expert_policy = ExpertPolicy()
# print(f"Training DQN with Expert Actions for 5000 Steps...")
# dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100, target_model_update=1e-2, policy=expert_policy)
# dqn.compile(adam_v2.Adam(learning_rate=lr), metrics=['mae'])
# dqn.fit(env, nb_steps=5000, verbose=2)
# print('Saving DQN Expert Weights...')
# dqn.save_weights('training/saved_models/DQN/BC/maze%d/dqn_bc_weights.h5f' % sample_maze, overwrite = True)

# # train
# print(f"Training DQN for {bc_dqn_learn} training steps...")
# policy = EpsGreedyQPolicy()
# # policy = GreedyQPolicy() # slower
# dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
# dqn.compile(adam_v2.Adam(learning_rate=lr), metrics=['mae'])
# print('Load in DQN Expert Weights...')
# dqn.load_weights('training/saved_models/DQN/BC/maze%d/dqn_bc_weights.h5f' % sample_maze)
# dqn.fit(env, nb_steps=bc_dqn_learn, verbose=2)
# dqn.save_weights('training/saved_models/DQN/BC/maze%d/dqn_bc_trained_eps_weights.h5f' % sample_maze, overwrite = True)

# load
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
dqn.compile(adam_v2.Adam(learning_rate=lr), metrics=['mae'])
dqn.load_weights('training/saved_models/DQN/BC/maze%d/dqn_bc_trained_eps_weights.h5f' % sample_maze)
print(f'Loading DQN {dqn} after training for only {bc_dqn_learn} steps...')

# evaluate
cb_ep = EpisodeLogger()
dqn.test(env, nb_episodes=5, visualize=True, callbacks=[cb_ep])
dqn_results(cb_ep, True)
