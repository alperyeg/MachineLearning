# Based on https://github.com/parvkpr/Simple-A2C-Pytorch-MountainCarv0/blob/master/doodle.py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import seaborn as sns

from collections import namedtuple
from torch.distributions import Categorical

sns.set(style="white",
        rc={'xtick.bottom': True, 'ytick.left': True})
sns.set_color_codes("dark")
sns.set_context("paper", font_scale=1.3,
                rc={"lines.linewidth": 1.8, "grid.linewidth": 0.1})


# define constant
GAMMA = 0.99
EPSILON = 0.99
EPISODES = 1000
LOG_INTERVAL = 10
SEED = 0
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


def epsilon_value(epsilon):
    return 0.99 * epsilon


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        # Defines the main part, i.e. the basis layers of the network
        self.basis_layers = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # actor head
        self.actor = nn.Linear(64, num_actions)
        # critic head
        self.critic = nn.Linear(64, 1)
        self.action_history = []
        self.rewards_achieved = []

    def forward(self, states):
        x = self.basis_layers(states)
        return self.critic(x), x

    def act(self, states, eps):
        val, x = self(states)
        x = F.softmax(self.actor(x), dim=-1)
        # categorical creates an array from a probability distribution,
        # the position with of a higher probability values is returned
        # here the probability is the action to take
        m = Categorical(x)
        # greedy exploration
        e_greedy = random.random()
        if e_greedy > eps:
            do_action = m.sample()
        else:
            # sample 3 values from the probability distribution
            # and pick a random one
            do_action = m.sample((3,))
            do_action = do_action[random.randint(-1, 2)]
        return val, do_action, m.log_prob(do_action)


def perform_updates(network, optm):
    """
    Update the Actor and Critic network parameters
    """
    r = 0
    saved_action = network.action_history
    rewards = []
    policy_losses = []
    critic_losses = []
    # go backwards in the rewards
    # and create queue of rewards,
    # putting the reward as the first element
    for i in network.rewards_achieved[::-1]:
        # this is part of the bellman step
        r = GAMMA * r + i
        rewards.insert(0, r)
    returns = torch.tensor(rewards)

    for (log_prob, val), R in zip(saved_action, returns):
        # calculate the advantage Q(s,a) - Q(s', a')
        advantage = R - val.item()
        # policy loss
        policy_losses.append(-log_prob * advantage)
        # value loss
        critic_losses.append(F.mse_loss(val, torch.tensor([R])))
    optm.zero_grad()

    # cumulative loss
    loss = torch.stack(policy_losses).sum() + torch.stack(critic_losses).sum()
    loss.backward()
    optm.step()
    # clear action history and rewards for next episode
    model.rewards_achieved.clear()
    model.action_history.clear()
    return loss.item()


if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    env.reset(seed=SEED)
    torch.manual_seed(SEED)
    n_inputs = 4
    eps = epsilon_value(EPSILON)
    losses = []
    counters = []
    plot_rewards = []
    model = ActorCritic(num_inputs=n_inputs,
                        num_actions=env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for idx in range(EPISODES):
        print(f'Index of episode {idx}')
        counter = 0
        state, _ = env.reset()
        ep_reward = 0
        loss = np.inf
        done = False
        truncated = False

        while not done:
            # unrolling state and getting action from the nn output
            state = torch.from_numpy(state).float()
            value, action, ac_log_prob = model.act(state, eps)
            model.action_history.append(SavedAction(ac_log_prob, value))
            # Agent action step
            # next_obs, reward, is_done, truncated, _
            state, reward, done, truncated, _ = env.step(action.item())

            model.rewards_achieved.append(reward)
            ep_reward += reward
            counter += 1
            # if counter % 5 == 0:
            loss = perform_updates(model, optimizer)
            # decay epsilon after each episode with rate 0.99
            eps = epsilon_value(eps)

        # save the losses
        if idx % LOG_INTERVAL == 0:
            losses.append(loss)
            counters.append(counter)
            plot_rewards.append(ep_reward)

    # plotting loss
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.savefig('loss1_cartpole.png')

    # plotting number of timesteps elapsed before convergence
    plt.clf()
    plt.xlabel('Episodes')
    plt.ylabel('timesteps')
    plt.plot(counters)
    plt.savefig('timestep_cartpole.png')
    # plotting total rewards achieved during all episodes
    plt.clf()
    plt.xlabel('Episodes')
    plt.ylabel('rewards')
    plt.plot(plot_rewards)
    plt.savefig('rewards_cartpole.png')
    plt.ion()
    fig, ax = plt.subplots(1)
    for i in range(200):
        state, _ = env.reset()
        state = torch.from_numpy(state).float()
        value, action, ac_log_prob = model.act(state, eps)
        observation, reward, terminated, truncated, info = env.step(action.item())
        text = ''
        if i % 4 == 0:
            r = env.render()
            if action.item() == 0:
                text = 'Pushing left'
            elif action.item() == 1:
                text = 'Pushing right'
            ims = ax.imshow(r)
            txt = ax.text(25, 30, f'{text}', bbox={'facecolor': 'white'})
            fig.canvas.draw()
            fig.canvas.flush_events()
            txt.set_visible(False)
