import random
import pdb

import gym
import numpy as np

env = gym.make('Pendulum-v0')


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
        if random.random() < 0.9:
            s = 1
        else:
            s = -1

        return np.array([s * w / abs(w) * self.t])


class Model():
    def __init__(self, in_size):
        '''
        Initialize the model to use to approximate the value function. This can be extended
        further to complicatted models. Currently a simple linear layer is used for the purposes
        of the assignment cited above.

        in_size: the size of the state features.
        '''
        self.in_size = in_size
        self.weights = np.empty([in_size, 1])

        self.reset_weights()

    def reset_weights(self):
        self.weights = np.random.rand(self.in_size, 1) * (0.002) - 0.001

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


# TODO: Change this
inp_size = 500
alpha = 0.1

pol = Policy(0.9, 1)
disp_vector = np.array([1, 1])
tile = TileCoding(5, 10, disp_vector)
model = Model(inp_size)
ret = ReturnCalculator()

for _ in range(200):
    # TODO: Change this to initialize the env from (0, 0) every episode
    state = env.reset()
    done = False

    # episode
    while done is False:
        action = pol.get_action(state[2])
        state, reward, done, info = env.step(action)
        state_feat = tile.get_features(state)
        pdb.set_trace()
        val = model.forward(state_feat)
        # target = ret.get_return()

        # model.weights = model.weights + (alpha / tile.n_tiles) * (target - val) * state_feat
