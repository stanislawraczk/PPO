from collections import deque
from time import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch

from ppo import Agent, TrajectoryBuffer, Logger
from global_variables import *

# env = gym.make('Humanoid-v4', render_mode='human')
# env = gym.make("BipedalWalker-v3", hardcore=True)
env = gym.make('Walker2d-v4', render_mode='rgb_array')
# env = gym.make('MountainCarContinuous-v0')
# env = gym.make("Pendulum-v1", render_mode="human")
env = gym.wrappers.RecordVideo(
    env,
    "videos_tests/",
    episode_trigger=lambda t: t % 100 == 0,
)

action_size = env.action_space.shape[0]
state_size = env.observation_space.shape[0]

start_timestamp = datetime.now().strftime("%d-%m-%Y-%H%M%S")

logger = Logger(time(), start_timestamp)
agent = Agent(state_size, action_size, logger, INITIAL_LOGSTD_SCALING)
buffer = TrajectoryBuffer(state_size, action_size, logger, agent, env)

agent.actor_network.load_state_dict(torch.load('checkpoints\checkpoint_actor_model_ppo.pth'))
agent.critic_network.load_state_dict(torch.load('checkpoints\checkpoint_critic_model_ppo.pth'))

num_episodes = 10
scores = []
episode_scores_window = deque(maxlen=100)
episode_scores = []
for i_episode in range(num_episodes):
    scores_per_episode = []
    state = env.reset()[0]
    while True:
        action, _ = agent.act(state, train=False)
        state, reward, done, truncated, info = env.step(action)
        scores_per_episode.append(reward)
        if done or truncated:
            break

    scores += scores_per_episode
    episode_scores_window.append(np.sum(scores_per_episode))
    episode_scores.append(np.sum(scores_per_episode))
    print(
        f"episode: {i_episode}; score: {np.sum(scores_per_episode)}; average score: {np.mean(episode_scores_window)}"
    )
