import random

class Environment:
    def __init__(self):
        self.steps_left = 10

    def get_observations(self):
        return [0., 0., 0.]

    def get_action(self):
        return [0, 1]

    def is_done(self):
        return self.steps_left == 0

    def actions(self, action):
        if self.is_done():
            # raise Exception('Game done')
            print(f'Game is done with action {action}')
            exit()
        self.steps_left -= 1
        return random.random()


class Agent:
    def __init__(self):
        self.total_reward = 0.

    def step(self, env:Environment):
        print(f'Total reward {self.total_reward}')
        current_obs = env.get_observations()
        print('obs: ', current_obs)
        actions = env.get_action()
        print('actions: ', actions)
        reward = env.actions(random.choice(actions))
        self.total_reward += reward

if __name__ == "__main__":
    env = Environment()
    agent = Agent()

    while not env.is_done():
        agent.step(env)

    print("Total reward got: %.4f" % agent.total_reward)
