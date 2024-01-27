from collections import deque
from time import time
from datetime import datetime

import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import torch

from ppo import Agent, TrajectoryBuffer, Logger

env = gym.make('Humanoid-v4', render_mode='rgb_array')
# env = gym.make("BipedalWalker-v3", hardcore=True)
# env = gym.make('MountainCarContinuous-v0')
# env = gym.make("Pendulum-v1", render_mode='rgb_array')
env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.RecordVideo(
    env,
    "videos/",
    episode_trigger=lambda t: t % 100 == 0,
)

action_size = env.action_space.shape[0]
state_size = env.observation_space.shape[0]

start_timestamp = datetime.now().strftime("%d-%m-%Y-%H%M%S")

logger = Logger(time(), start_timestamp)
agent = Agent(state_size, action_size, logger)
buffer = TrajectoryBuffer(state_size, action_size, logger, agent, env)

# agent.actor_network.load_state_dict(torch.load(r'checkpoints\best_actor_model_ppo.pth'))
# agent.critic_network.load_state_dict(torch.load(r'checkpoints\best_critic_model_ppo.pth'))

num_episodes = 50001

buffer.collect_trajectories(num_episodes)

logger.log_finish(time())
