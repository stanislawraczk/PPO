from collections import deque
from time import time

import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import torch

from ppo import Agent, TrajectoryBuffer, Logger

# env = gym.make('Humanoid-v4', render_mode='human')
# env = gym.make("BipedalWalker-v3", hardcore=True)
# env = gym.make('MountainCarContinuous-v0')
env = gym.make("Pendulum-v1", render_mode='rgb_array')
env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.RecordVideo(
    env,
    "videos/",
    episode_trigger=lambda t: t % 100 == 0,
)

action_size = env.action_space.shape[0]
state_size = env.observation_space.shape[0]

logger = Logger(time())
agent = Agent(state_size, action_size, logger)
buffer = TrajectoryBuffer(state_size, action_size, logger, agent, env)

num_episodes = 5001

buffer.collect_trajectories(num_episodes)
