import sys
from collections import deque

from time import time

import torch
from torch import FloatTensor
from torch.nn import functional as F
from torch import optim
import numpy as np
from matplotlib import pyplot as plt

from model_ppo import ActorPPO, CriticPPO

from global_variables import *

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class Agent:
    def __init__(self, state_size, action_size, logger):
        self.state_size = state_size
        self.action_size = action_size
        self.actor_network = ActorPPO(state_size, action_size).to(device)
        self.critic_network = CriticPPO(state_size).to(device)

        self.actor_network_optim = optim.Adam(
            self.actor_network.parameters(),
            lr=LR_ACTOR,
            weight_decay=WEIGHTS_DECAY_ACTOR,
        )
        self.critic_network_optim = optim.Adam(
            self.critic_network.parameters(),
            lr=LR_CRTITC,
            weight_decay=WEIGHTS_DECAY_CRITIC,
        )

        self.entropy_coef = torch.tensor(ENTROPY_COEF).float().to(device)
        self.std = SD

        self.logger = logger

    def act(self, state):
        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            # mu, sigma = self.actor_network(state)
            mu = self.actor_network(state)
        dist = torch.distributions.Normal(mu, self.std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action = action.cpu().data.numpy()
        log_prob = log_prob.cpu().data.numpy()
        return action, log_prob

    def evaluate(self, state):
        state = torch.tensor(state).float().to(device)
        with torch.no_grad():
            value = self.critic_network(state).squeeze()

        value = value.cpu().data.numpy()
        return value

    def learn(self, batch):
        states, actions, log_probs, advantages, discounted_rewards = batch

        advantages = (advantages - advantages.mean()) / advantages.std()

        # mus, sigmas = self.actor_network(states)
        mus = self.actor_network(states)
        expected_values = self.critic_network(states).squeeze()
        dists = torch.distributions.Normal(mus, self.std)

        curr_log_probs = dists.log_prob(actions)

        entropy_loss = dists.entropy().mean()

        value_function_loss = torch.nn.MSELoss()(expected_values, discounted_rewards)

        probability_ratios = torch.exp(curr_log_probs - log_probs).squeeze()
        probability_ratios_clipped = torch.clip(
            probability_ratios, min=1 - CLIP_COEF, max=1 + CLIP_COEF
        )
        actor_loss_clipped = torch.min(
            probability_ratios_clipped * advantages, probability_ratios * advantages
        )
        # actor_loss_final = -(actor_loss_clipped + self.entropy_coef * entropy_loss).mean()
        actor_loss_final = -actor_loss_clipped.mean()

        self.actor_network_optim.zero_grad()
        actor_loss_final.backward()
        self.actor_network_optim.step()

        self.critic_network_optim.zero_grad()
        value_function_loss.backward()
        self.critic_network_optim.step()

        self.logger.log_losses(
            actor_loss_final.cpu().data.numpy(), value_function_loss.cpu().data.numpy()
        )


class TrajectoryBuffer:
    def __init__(self, state_size, action_size, logger, agent, env) -> None:
        self.max_episode_length = MAX_EPISODE_LENGTH
        self.number_of_timesteps = NUMBER_OF_TIMESTEPS
        self.action_size = action_size
        self.state_size = state_size
        self.logger = logger
        self.agent = agent
        self.env = env
        self.clear_memory()

    def collect_trajectories(self, num_episodes: int) -> (bool, float):
        for episode_num in range(num_episodes):
            state = self.env.reset()[0]
            episode_rewards = []
            for i in range(self.max_episode_length):
                self.states.append(state)
                action, log_prob = self.agent.act(state)

                state, reward, done, truncated, _ = self.env.step(action)

                self.actions.append(action)
                self.log_probs.append(log_prob)
                episode_rewards.append(reward)

                if done or truncated:
                    break

            deltas = self.calculate_deltas(episode_rewards, last_state=state)
            advantages = self.calculate_advantages(deltas)
            self.advantages.extend(advantages)

            discounted_rewards = self.calculate_discounted_rewards(episode_rewards)
            self.discounted_rewards.extend(discounted_rewards)

            self.logger.log_rewards(episode_rewards, sum(episode_rewards))
            if episode_num % 10 == 0:
                self.logger._log_summary(episode_num, time())

            self.steps_in_memory += i

            if self.steps_in_memory >= self.number_of_timesteps:
                for _ in range(NUM_EPOCHS):
                    batch = self.batch()
                    self.agent.learn(batch)
                self.clear_memory()
                
            if episode_num % CHECKPOINT_EVERY == 0:
                torch.save(self.agent.actor_network.state_dict(), "checkpoints\checkpoint_actor_model_ppo.pth")
                torch.save(self.agent.critic_network.state_dict(), "checkpoints\checkpoint_critic_model_ppo.pth")

    def calculate_deltas(self, rewards, last_state):
        trajectory_len = len(rewards)
        states = self.states[-trajectory_len:]
        states.append(last_state)
        deltas = []
        for idx, reward in enumerate(rewards):
            value = self.agent.evaluate(states[idx])
            next_value = self.agent.evaluate(states[idx + 1])
            delta = reward + GAMMA * next_value - value
            deltas.append(delta)
        return deltas

    def calculate_advantages(self, deltas):
        advantages = []
        advantage = 0
        for delta in reversed(deltas):
            advantage = delta + advantage * GAMMA * LAMBDA
            advantages.insert(0, advantage)
        return advantages

    def calculate_discounted_rewards(self, rewards):
        discounted_rewards = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + discounted_reward * GAMMA
            discounted_rewards.insert(0, discounted_reward)
        return discounted_rewards

    def batch(self):
        states = torch.tensor(self.states).float().to(device)
        actions = torch.tensor(self.actions).float().to(device)
        log_probs = torch.tensor(self.log_probs).float().to(device)
        advantages = torch.tensor(self.advantages).float().to(device)
        discounted_rewards = torch.tensor(self.discounted_rewards).float().to(device)

        return states, actions, log_probs, advantages, discounted_rewards

    def clear_memory(self):
        self.steps_in_memory = 0

        self.states = []
        self.actions = []
        self.log_probs = []
        self.advantages = []
        self.discounted_rewards = []


class Logger:
    def __init__(self, start_time) -> None:
        self.rewards = []
        self.episode_returns = []
        self.episode_returns_window = deque(maxlen=100)
        self.actor_loss = []
        self.critic_loss = []
        self.start_time = start_time

    def _log_summary(self, episode_num, stop_time):
        print(f"========== Episode #{episode_num}==========")
        print(
            f"Average Episodic Return in last 100 episodes: {round(np.mean(self.episode_returns_window), 3)}"
        )
        print(f"This episode return: {round(self.episode_returns[-1], 3)}")
        time_measure = stop_time - self.start_time
        self.start_time = stop_time
        print(f"Time for last 10 episodes: {round(time_measure, 3)} seconds")

        if episode_num % 100 == 0:
            self._save_graphs(episode_num)

    def log_rewards(self, rewards, episode_return):
        self.rewards.extend(rewards)
        self.episode_returns.append(episode_return)
        self.episode_returns_window.append(episode_return)

    def log_losses(self, actor_loss, critic_loss):
        self.actor_loss.append(actor_loss)
        self.critic_loss.append(critic_loss)

    def _save_graphs(self, episode_num):
        moving_average_episode_returns = []
        moving_average_window = 50
        for i in range(len(self.episode_returns) - moving_average_window + 1):
            moving_average = np.mean(self.episode_returns[i:i+moving_average_window])
            moving_average_episode_returns.append(moving_average)

        plt.plot(moving_average_episode_returns)
        plt.title("episode scores")
        plt.savefig(f"plots/episode_scores_{episode_num}.png")
        plt.close()
        plt.plot(self.actor_loss)
        plt.title("actor loss")
        plt.savefig(f"plots/actor_loss_{episode_num}.png")
        plt.close()
        plt.plot(self.critic_loss)
        plt.title("critic loss")
        plt.savefig(f"plots/critic_loss_{episode_num}.png")
        plt.close()
