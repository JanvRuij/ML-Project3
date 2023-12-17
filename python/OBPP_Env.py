# Create the knapsack enviroment
import random
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

# Parameters
# capacity of the bins
C = 200
# number of items until we see new items
M = 10
# number of items in total
N = 200
# max size of each item
S = 100


class BalanceEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __init__(self):
        self.seed()
        self.state = self.reset()
        # place item (1 to M) into a bin (1 to N)
        self.action_space = spaces.Discrete(M, N)
        self.observation_space = spaces.Box(
            low=np.array([0.0] * (4 + M + N)),
            high=np.array([N] + [M] + [C] + [M] + [S]*M + [S]*N),
            dtype=np.uint32)

    def reset(self):
        self.C = C
        self.m = M
        self.generated = 0
        # create random instance from 150 to 200 items
        self.nbItems = random.randint(150, 200)
        self.x = np.zeros((self.nbItems, self.nbItems))
        # create random items
        self.items = np.random.randint(1, S, self.nbItems)
        desired_shape = (1, N)
        padding = desired_shape[1] - self.items.shape[0]
        self.items = np.pad(self.items, (0, padding), "constant", constant_values=0)
        self.visible_items = self.items[:self.m]
        self.greedy_value = self._greedy()
        self.total_reward = 0
        self.total_weight = 0
        self.state = self._update_state()
        return self.state

    def step(self, action_item, action_bin):
        reward = 0
        done = False
        self.total_weight += np.sum(self.x, axis=1)
        if self.visible_items[action_item] == 0:
            reward = 0
            done = True
        elif np.sum(self.x[action_bin], axis=1) + self.visible_items[action_item] > self.C:
            reward = 0
            done = True
        else:
            rowsum = np.sum(self.x, axis=1)
            self.x = np.zeros((self.nbItems, self.nbItems))
            nr_bins = np.count_nonzero(rowsum, axis=0)
            # the less bins the higher the reward
            reward = N - nr_bins
            self.visible_items[action_item] = 0
            if np.sum(self.visible_items) == 0 and self.generated < self.nbItems:
                self.generate_new_items()

        self.total_reward += reward
        self.state = self._update_state()

        return self.state, reward, done, {}

    def _update_state(self):
        total_weights = np.sum(self.x, axis=0)
        tempS = np.array([self.nbItems, M, C, self.generated, S])

        state = np.concatenate((tempS, self.visible_items, total_weights))
        return state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(
                round(self.total_reward, 3), "(",
                round(self.total_reward - self.greedy, 3), ")",
                round(self.total_weight/self.C, 3)
            )

    def generate_new_items(self):
        self.visible_items = self.items[self.generated:self.generated + self.m]
        self.generated += self.m

    def _greedy(self):
        self.visible_items.sort()
        for item in self.visible_items:
            # sort the indexes from high row sum to low
            sorted_indices = np.flip(np.argsort(self.x.sum(axis=1), axis=0))
            # we want to append the item to the bin with the most items
            for idx in sorted_indices:
                # check if the item fits
                if np.sum(self.x[idx]) + item <= self.C:
                    # if it fits we append it to the first open position
                    zero_index = np.argwhere(self.x[idx] == 0)[0]
                    self.x[idx][zero_index] = item
                    break

        # if we havent generated n items we continue the process
        if self.generated < self.nbItems:
            self.generate_new_items()
            return self._greedy()

        # otherwise we delete all empty bins and return (for readibility)
        else:
            rowsum = np.sum(self.x, axis=1)
            # reset everything for the neural network to use
            self.x = np.zeros((self.nbItems, self.nbItems))
            self.visible_items = self.items[:self.m]
            nr_bins = np.count_nonzero(rowsum, axis=0)
            return nr_bins


x = BalanceEnv()
