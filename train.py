from collections import deque
from time import time
from datetime import datetime
import random

import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import torch

from ppo import Agent, TrajectoryBuffer, Logger
from global_variables import *

# env = gym.make('Humanoid-v4', render_mode='rgb_array')
env = gym.make('Walker2d-v4', render_mode='rgb_array')
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

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
_ = env.reset(seed=SEED)

start_timestamp = datetime.now().strftime("%d-%m-%Y-%H%M%S")

logger = Logger(time(), start_timestamp)
agent = Agent(state_size, action_size, logger, INITIAL_LOGSTD_SCALING)
buffer = TrajectoryBuffer(state_size, action_size, logger, agent, env)

# agent.actor_network.load_state_dict(torch.load(r'checkpoints\checkpoint_actor_model_ppo.pth'))
# agent.critic_network.load_state_dict(torch.load(r'checkpoints\checkpoint_critic_model_ppo.pth'))

num_timesteps = 1_000_000

buffer.collect_trajectories(num_timesteps)

logger.log_finish(time())
