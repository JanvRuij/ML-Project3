import numpy as np


class OnlineBPP:
    def __init__(self, n, m) -> None:
        # number of items
        self.n = n
        # number of items after which the new items are generated
        self.m = m
        # generate m random items
        self.items = np.random.rand(m)
        # capacity of the bins
        self.c = 20
        # solution vector
        self.x = np.zeros((n, n))
        # to keep track of items generated
        self.generated = m

    def generate_new_items(self):
        self.items = np.random.rand(self.m)
        self.generated += self.m

    def greedy(self):
        self.items.sort()
        for item in self.items:
            # sort the indexes from high row sum to low
            sorted_indices = np.argsort(self.x.sum(axis=1))[::-1]
            # we want to append the item to the bin with the most items
            for idx in sorted_indices:
                # check if the item fits
                if np.sum(self.x[idx]) + item <= self.c:
                    # if it fits we append it
                    non_zero_index = np.argmin(self.x[idx] != 0)
                    self.x[idx][non_zero_index] = item
                    break

        # if we havent generated n items we continou the process
        if self.generated < self.n:
            self.generate_new_items()
            self.greedy()

        else:
            self.x = self.x[~np.all(self.x == 0, axis=1)]
            return


x = OnlineBPP(100, 10)
x.greedy()
print(x.x)
