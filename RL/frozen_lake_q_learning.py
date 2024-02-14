#!/usr/bin/env python3

# from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
import collections

import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

GAMMA = 0.9
TEST_EPISODES = 20

class Agent:
    def __init__(self) -> None:
        self.env = gym.make("FrozenLake-v1")
        self.state, self.info = self.env.reset()
        #  A dictionary with the composite key "source state" + "action" + "target state".
        #  The value is obtained from the immediate reward
        self.rewards = collections.defaultdict(float)
        #  A dictionary keeping counters of the experienced transitions.
        # The key is the composite "state" + "action" and the value is another dictionary
        # that maps the target state in to a count of times.
        self.transits = collections.defaultdict(collections.Counter)
        # A dictionary that maps a state into the calculated value of this state.
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, truncated, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset()[0] if is_done else new_state

    # def calc_action_value(self, state, action):
    #     target_counts = self.transits[(state, action)]
    #     total = sum(target_counts.values())
    #     action_value = 0.0
    #     for tgt_state, count in target_counts.items():
    #         reward = self.rewards[(state, action, tgt_state)]
    #         #  calculation of the state's value using Bellman equation
    #         #  i.e. prob * reward + gamma * V_s'
    #         action_value += (count / total) * (reward + GAMMA * self.values[tgt_state])
    #     return action_value

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episodes(self, env):
        total_reward = 0.0
        state, _ = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, truncated, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transits[(state, action)]
                total = sum(target_counts.values())
                for tgt_state, count in target_counts.items():
                    reward = self.rewards[(state, action, tgt_state)]
                    best_action = self.select_action(tgt_state)
                    action_value += (count / total) * (reward + GAMMA * self.values[(tgt_state, best_action)])
                self.values[(state, action)] = action_value


if __name__ == '__main__':
    test_env = gym.make("FrozenLake-v1")
    agent = Agent()
    # writer = SummaryWriter(comment="-v-iteration")
    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episodes(test_env)
        reward /= TEST_EPISODES
        # writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print(f"Best reward updates {best_reward} -> {reward}")
        if reward > 0.80:
            print(f"Solved in {iter_no} iterations")
            break
    # writer.close()
