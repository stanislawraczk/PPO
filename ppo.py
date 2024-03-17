from collections import deque

from time import time

import torch
from torch import optim
import numpy as np
from matplotlib import pyplot as plt

from model_ppo import ActorPPO, CriticPPO

from global_variables import *

if CUDA:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


class Agent:
    def __init__(self, state_size, action_size, logger, initial_logstd_scaling):
        self.state_size = state_size
        self.action_size = action_size
        self.actor_network = ActorPPO(state_size, action_size, initial_logstd_scaling=initial_logstd_scaling).to(device)
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

        self.logger = logger

    def act(self, state, train=True):
        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            mu = self.actor_network(state)
            logstd = self.actor_network.logstd.expand_as(mu)
            std = torch.exp(logstd)
            self.logger.log_std(std.cpu().data.numpy())
        if train:
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            log_prob = torch.sum(dist.log_prob(action))
            action = action.cpu().data.numpy()
            log_prob = log_prob.cpu().data
        else:
            action = mu.cpu().data.numpy()
            log_prob = None
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

        mus = self.actor_network(states)
        logstd = self.actor_network.logstd.expand_as(mus)
        stds = torch.exp(logstd)
        expected_values = self.critic_network(states).squeeze()
        try:
            dists = torch.distributions.Normal(mus, stds)
        except ValueError as err:
            print(err)
            self.logger.save_graphs()
            self.logger.save_logs('Episode with error')
            raise ValueError

        curr_log_probs = torch.sum(dists.log_prob(actions), dim=1)

        entropy_loss = dists.entropy().mean()

        value_function_loss = torch.nn.MSELoss()(expected_values, discounted_rewards)

        probability_ratios = torch.exp(curr_log_probs - log_probs).squeeze()
        probability_ratios_clipped = torch.clip(
            probability_ratios, min=1 - CLIP_COEF, max=1 + CLIP_COEF
        )
        actor_loss_clipped = torch.min(
            probability_ratios_clipped * advantages, probability_ratios * advantages
        )
        actor_loss_final = -(actor_loss_clipped + self.entropy_coef * entropy_loss).mean()

        self.actor_network_optim.zero_grad()
        actor_loss_final.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), .5)
        self.actor_network_optim.step()

        self.critic_network_optim.zero_grad()
        value_function_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), .5)
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

    def collect_trajectories(self, n_timesteps: int) -> None:
        n_steps = 0
        episode_num = 0
        while n_steps < n_timesteps:
            state = self.env.reset()[0]
            episode_rewards = []
            for i in range(self.max_episode_length):
                n_steps += 1
                self.states.append(state)
                action, log_prob = self.agent.act(state)

                state, reward, done, truncated, _ = self.env.step(action)

                self.actions.append(action)
                self.rewards.append(reward)
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
                self.logger._log_summary(episode_num, time(), n_steps)

            self.steps_in_memory += i

            if self.steps_in_memory >= self.number_of_timesteps:
                for _ in range(NUM_EPOCHS):
                    mini_batches = self.batch()
                    for mini_batch in mini_batches:
                        self.agent.learn(mini_batch)
                self.clear_memory()
                
            if episode_num % CHECKPOINT_EVERY == 0:
                torch.save(self.agent.actor_network.state_dict(), "checkpoints\checkpoint_actor_model_ppo.pth")
                torch.save(self.agent.critic_network.state_dict(), "checkpoints\checkpoint_critic_model_ppo.pth")
            episode_num += 1

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
        batch_len = len(self.states)
        shuffle_indexes = np.arange(batch_len)
        mini_batches_num = batch_len // BATCH_SIZE
        mini_batches = []
        np.random.shuffle(shuffle_indexes)

        states = np.array(self.states)[shuffle_indexes]
        actions = np.array(self.actions)[shuffle_indexes]
        log_probs = np.array(self.log_probs)[shuffle_indexes]
        advantages = np.array(self.advantages)[shuffle_indexes]
        discounted_rewards = np.array(self.discounted_rewards)[shuffle_indexes]

        states = torch.tensor(states).float().to(device)
        actions = torch.tensor(actions).float().to(device)
        log_probs = torch.tensor(log_probs).float().to(device)
        advantages = torch.tensor(advantages).float().to(device)
        discounted_rewards = torch.tensor(discounted_rewards).float().to(device)

        for i in range(mini_batches_num-1):
            mini_batch_states = states[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
            mini_batch_actions = actions[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
            mini_batch_log_probs = log_probs[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
            mini_batch_advantages = advantages[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
            mini_batch_rewards = discounted_rewards[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]

            mini_batches.append((mini_batch_states, mini_batch_actions, mini_batch_log_probs, mini_batch_advantages, mini_batch_rewards))

        mini_batch_states = states[(mini_batches_num-1)*BATCH_SIZE:]
        mini_batch_actions = actions[(mini_batches_num-1)*BATCH_SIZE:]
        mini_batch_log_probs = log_probs[(mini_batches_num-1)*BATCH_SIZE:]
        mini_batch_advantages = advantages[(mini_batches_num-1)*BATCH_SIZE:]
        mini_batch_rewards = discounted_rewards[(mini_batches_num-1)*BATCH_SIZE:]

        mini_batches.append((mini_batch_states, mini_batch_actions, mini_batch_log_probs, mini_batch_advantages, mini_batch_rewards))

        return mini_batches

    def clear_memory(self):
        self.steps_in_memory = 0

        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.advantages = []
        self.discounted_rewards = []


class Logger:
    def __init__(self, start_time, start_timestamp) -> None:
        self.rewards = []
        self.episode_returns = []
        self.episode_returns_window = deque(maxlen=100)
        self.actor_loss = []
        self.critic_loss = []
        self.std = []
        self.state_mean = []
        self.state_std = []
        self.loss_logged = 0
        self.std_logged = 0
        self.episode_logged = 0
        self.episde_start_time = start_time
        self.start_time = start_time
        self.start_timestamp = start_timestamp
        self.log_file_path = 'logs/log_' + self.start_timestamp + '.txt'

    def _log_summary(self, episode_num, stop_time, n_steps):
        print(f"==================== Episode #{episode_num}         ====================")
        print(f"==================== Number of timesteps #{n_steps} ====================")
        print(
            f"Average Episodic Return in last 100 episodes: {round(np.mean(self.episode_returns_window), 3)}"
        )
        print(f"This episode return: {round(self.episode_returns[-1], 3)}")
        time_measure = stop_time - self.episde_start_time
        self.episde_start_time = stop_time
        print(f"Time for last 10 episodes: {round(time_measure, 3)} seconds")

        if episode_num % 100 == 0:
            self.save_graphs(n_steps)
            self.save_logs(episode_num)

    def log_state_stats(self, mean, std):
        self.state_mean.append(np.copy(mean))
        self.state_std.append(np.copy(std))

    def log_rewards(self, rewards, episode_return):
        self.rewards.extend(rewards)
        self.episode_returns.append(episode_return)
        self.episode_returns_window.append(episode_return)
        self.episode_logged += 1

    def log_losses(self, actor_loss, critic_loss):
        self.actor_loss.append(actor_loss)
        self.critic_loss.append(critic_loss)
        self.loss_logged += 1

    def log_std(self, std):
        self.std.append(std)
        self.std_logged += 1

    def save_logs(self, episode_num):
        episode_returns = [f'{r:.2f}' for r in self.episode_returns[-self.episode_logged:]]
        insert_new_lines = range(10,self.episode_logged,10)
        for idx in insert_new_lines:
            episode_returns.insert(idx, '\n')
        episode_returns = ', '.join(episode_returns)
        actor_losses = [str(r) for r in self.actor_loss[-self.loss_logged:]]
        actor_losses = ', '.join(actor_losses)
        critic_losses = [str(r) for r in self.critic_loss[-self.loss_logged:]]
        critic_losses = ', '.join(critic_losses)
        with open(self.log_file_path, 'a') as f:
            f.write(f'==========Episode #{episode_num}==========\n')
            f.write(f'==========Episode Returns==========\n')
            f.write(episode_returns)
            f.write('\n')
            f.write(f'==========Actor Losses==========\n')
            f.write(actor_losses)
            f.write('\n')
            f.write(f'==========Critic Losses==========\n')
            f.write(critic_losses)
            f.write('\n')

        self.loss_logged = 0
        self.std_logged = 0
        self.episode_logged = 0

    def save_graphs(self, n_steps):
        moving_average_episode_returns = []
        moving_average_window = 50
        for i in range(len(self.episode_returns) - moving_average_window + 1):
            moving_average = np.mean(self.episode_returns[i:i+moving_average_window])
            moving_average_episode_returns.append(moving_average)

        plt.plot(moving_average_episode_returns)
        plt.title("episode scores")
        plt.savefig(f"plots/episode_scores/episode_scores.png", dpi=300)
        plt.close()
        plt.plot(self.actor_loss)
        plt.title("actor loss")
        plt.savefig(f"plots/actor_loss/actor_loss.png", dpi=300)
        plt.close()
        plt.plot(self.critic_loss)
        plt.title("critic loss")
        plt.savefig(f"plots/critic_loss/critic_loss.png", dpi=300)
        plt.close()
        plt.plot(self.std)
        plt.title("std")
        plt.savefig(f"plots/stds/std.png", dpi=300)
        plt.close()
        plt.plot(self.state_mean)
        plt.title("state mean")
        plt.savefig(f"plots/state_mean/state_mean.png", dpi=300)
        plt.close()
        plt.plot(self.state_std)
        plt.title("state std")
        plt.savefig(f"plots/state_std/state_std.png", dpi=300)
        plt.close()

    def log_finish(self, finish_time):
        training_time = finish_time - self.start_time
        with open(self.log_file_path, 'a') as f:
            f.write(f'Training time: {training_time / 60:.2f} minutes')
