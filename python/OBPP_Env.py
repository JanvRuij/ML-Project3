# Create the knapsack enviroment
import random
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

# Parameters
# capacity of the bins
C = 50
# number of items until we see new items
M = 5
# number of items in total
N = 100
# max size of each item
S = 40


class BalanceEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __init__(self):
        self.seed()
        self.state = self.reset()
        # place item (1 to M) into a bin (1 to N)
        self.action_space = spaces.Discrete(M)
        self.observation_space = spaces.Box(
            low=np.array([0.0] * (3 + M + N)),
            high=np.array([N] + [N] + [C] + [S]*M + [S]*N),
            dtype=np.uint32)

    def reset(self):
        # capacity of the bins
        self.C = C
        # number of visible items
        self.m = M
        # maximum number of items
        self.n = N
        # how much we have seen
        self.generated = self.m
        self.nr_bins = 0
        # create random instance from 75 to 100 items
        self.nbItems = random.randint(75, self.n)
        self.x = np.zeros((self.n, self.n))
        # create random items
        self.items = np.random.randint(1, S, self.nbItems)
        # fill the end with 0's
        self.items = np.pad(self.items, (0, self.n - self.nbItems), "constant")
        # we can only see m items
        self.visible_items = self.items[:self.m]
        self.greedy = self._greedy(0)
        self.total_reward = 0
        
        # start out with bin weight equal to 0
        self.total_weight = np.zeros(self.n)
        self.state = self._update_state()
        return self.state

    def step(self, action):
        reward = 0
        done = False
        # if we add a an item with value 0 we dont get a reward
        if self.visible_items[action] == 0:
            done = True

        else:        # find the first fitting bin
            for index in np.ndindex(self.n):
                if self.total_weight[index] + self.visible_items[action] < self.C:
                    self.total_weight[index] += self.visible_items[action]
                    break

            # otherwise we add the item to the bin
            nr_bins = np.count_nonzero(self.total_weight, axis=0)
            if nr_bins > self.nr_bins:
                reward = 1
                self.nr_bins = nr_bins
            else:
                reward = 0.5

            self.visible_items[action] = 0

        if np.sum(self.visible_items) == 0 and self.generated < self.nbItems:
            self.generate_new_items()

        elif self.generated >= self.nbItems and np.sum(self.visible_items) == 0:
            done = True

        self.total_reward += reward
        self.state = self._update_state()
        return self.state, reward, done, {}

    def _update_state(self):
        # the state contains the numer of items, the amount we have seen and the capacity
        tempS = np.array([self.nbItems, self.generated, self.C])
        # and the items it can see
        state = np.concatenate((tempS, self.visible_items))
        # and the total weight in each bin
        state = np.concatenate((state, self.total_weight))
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

    def _greedy(self, reward):
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
                    # give similar result as the NN
                    rowsum = np.sum(self.x, axis=1)
                    nr_bins = np.count_nonzero(rowsum, axis=0)
                    self.x[idx][zero_index] = item
                    if nr_bins > self.nr_bins:
                        reward += 1
                        self.nr_bins = nr_bins
                    else:
                        reward += 0.5
                    break

        # if we havent generated n items we continue the process
        if self.generated < self.nbItems:
            self.generate_new_items()
            return self._greedy(reward)

        else:
            # calculate nr of bins and return
            rowsum = np.sum(self.x, axis=1)
            nr_bins = np.count_nonzero(rowsum, axis=0)
            # reset the visible items for the NN
            self.visible_items = self.items[:self.m]
            self.generated = self.m
            self.nr_bins = 0
            return reward


x = BalanceEnv()
