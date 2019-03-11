import random
import pdb
import argparse

import gym
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for weights initialization")
params = parser.parse_args()

env = gym.make('Pendulum-v0')

# monkey patching reset() to always start from (0, 0)


def get_new_reset(env):

    def reset():
        env.state = np.array([0, 0])
        env.last_u = None
        return env._get_obs()

    return reset


env.env.reset = get_new_reset(env.env)


# pdb.set_trace()


class Policy():
    def __init__(self, p, t):
        '''
        Initializes the fixed policy to evaluate.

        p: probability of giving torque in the same direction as the angular velocity
        t: magnitude of the torque to produce
        '''

        self.p = p
        self.t = t

    def get_action(self, w):
        '''
        Returns the action to take.

        w: Angular velocity of the pendulum
        '''
        if random.random() < self.p:
            s = 1
        else:
            s = -1

        return np.array([s * (w + 1e-9) / (abs(w) + 1e-9) * self.t])


class Model():
    def __init__(self, in_size):
        '''
        Initialize the model to use to approximate the value function. This can be extended
        further to complicatted models. Currently a simple linear layer is used for the purposes
        of the assignment cited above.

        in_size: the size of the state features.
        '''
        self.in_size = in_size
        self.weights = np.empty([in_size])

        self.reset_weights()

    def reset_weights(self, seed=params.seed):
        np.random.seed(seed)
        self.weights = np.random.rand(self.in_size) * (0.002) - 0.001

    def forward(self, inp):
        return np.dot(inp, self.weights)


class TileCoding():
    def __init__(self, n_tilings, n_tiles, disp_vector):
        '''
        Initialize tile coding configuration. Assumes the input is normalized.
        '''
        # self.tile_size = [(max_val - min_val) / n_tiles for (min_val, max_val) in zip(min_state, max_state)]
        self.tile_size = 1 / n_tiles
        self.delta = self.tile_size / n_tilings

        self.n_tilings = n_tilings
        self.n_tiles = n_tiles

        self.tilings = []

        for i in range(n_tilings):
            self.tilings.append(i * self.delta * disp_vector)

    def get_features(self, state):
        '''
        Get the features of a state according to tilee coding config defined.

        state: state to get the features of.
        '''
        # gettting theta and theta_dot
        # pdb.set_trace()
        theta = np.arctan2(state[1], state[0])
        theta_dot = state[2]

        # normalize
        theta = (theta + np.pi) / (2 * np.pi)
        theta_dot = (theta_dot + 8) / (2 * 8)

        feat = np.zeros(self.n_tilings * self.n_tiles * self.n_tiles)

        for i in range(self.n_tilings):
            x = (theta - self.tilings[i][0]) // self.tile_size
            y = (theta_dot - self.tilings[i][1]) // self.tile_size
            feat[int((i * self.n_tiles * self.n_tiles) + y * self.n_tiles + x)] = 1

        return feat


class ReturnCalculator():
    def __init__(self):
        '''
        Calculates the return.
        '''
        pass

    def get_return(self):
        '''
        Calculates the return.
        '''
        pass


def generate_episode(env, init_state, pol):
    '''
    Generates an episode.

    env: the environment to generate the episode from
    init_state: starting state of the episode
    pol: polic to generate the episode from
    '''

    states = [init_state]
    rewards = []

    done = False

    while done is False:
        action = pol.get_action(states[-1][2])
        state, reward, done, info = env.step(action)
        states.append(state)
        rewards.append(reward)

    return states, rewards


inp_size = 500
alpha_list = [0.25, 0.125, 0.0625]
gamma = 0.9
decay_factor_list = [0]

pol = Policy(0.9, 1)
disp_vector = np.array([1, 1])
tile = TileCoding(5, 10, disp_vector)
model = Model(inp_size)
ret = ReturnCalculator()

value = np.zeros((len(alpha_list), 200))

for l, decay_factor in enumerate(decay_factor_list):
    for k, alpha in enumerate(alpha_list):

        val = np.zeros((10, 200))

        for i in range(10):

            model.reset_weights(params.seed + i)

            for j in range(200):
                state = env.reset()
                state_feat = tile.get_features(state)
                val[i][j] = model.forward(state_feat)
                print('Run %d for lambda = %.2f, alpha = %.2f; Value = %.2f\r' % (i, decay_factor, alpha, val[i][j]), end="")
                trace = np.zeros(state_feat.shape)
                done = False

                # episode
                states, rewards = generate_episode(env, state, pol)
                returns = np.zeros(len(rewards) + 1)
                for idx in range(len(rewards)):
                    returns[-idx - 2] = (gamma * returns[-idx - 1] + rewards[-idx - 1])
                for state, ret in zip(states[:-1], returns[:-1]):
                    state_feat = tile.get_features(state)
                    delta = ret - model.forward(state_feat)
                    trace = gamma * decay_factor * trace + state_feat

                    model.weights = model.weights + (alpha / tile.n_tilings) * delta * trace

                # while done is False:
                #     action = pol.get_action(state[2])
                #     next_state, reward, done, info = env.step(action)
                #     next_state_feat = tile.get_features(next_state)
                #     delta = reward + gamma * model.forward(next_state_feat) - model.forward(state_feat)
                #     trace = gamma * decay_factor * trace + state_feat

                #     model.weights = model.weights + (alpha / tile.n_tilings) * delta * trace

                #     state = next_state
                #     state_feat = next_state_feat
        value[k] = np.mean(val, axis=0)
        print('Value of state (0, 0) for lambda = %.2f, alpha = %.2f = %.2f\n' % (decay_factor, alpha, value[k][-1]), end="")

    leg = []
    fig = plt.figure(figsize=(15, 6))
    for t, alpha in enumerate(alpha_list):
        plt.plot(value[t])
        leg = leg + [str(alpha)]
    plt.legend(leg)
    plt.title('Value of state (0, 0) with lambda %.2f' % decay_factor)

    plt.savefig('mc_l%.1f.png' % (decay_factor), dpi=fig.dpi)
