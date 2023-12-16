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

    # keep adding until it doesnt fit anymore (sorted first)
    def greedy1(self):
        self.items.sort()
        for item in self.items:
            # sort the indexes from high row sum to low
            sorted_indices = np.argsort(self.x.sum(axis=1))[::-1]
            # we want to append the item to the bin with the most items
            for idx in sorted_indices:
                # check if the item fits
                if np.sum(self.x[idx]) + item <= self.c:
                    # if it fits we append it
                    zero_index = np.argmin(self.x[idx] == 0)
                    self.x[idx][zero_index] = item
                    break

        # if we havent generated n items we continue the process
        if self.generated < self.n:
            self.generate_new_items()
            self.greedy1()

        # otherwise we delete all empty bins and return
        else:
            self.x = self.x[~np.all(self.x == 0, axis=1)]
            return

    # divide items evenly over two bins (also sorted)
    def greedy2(self):

        # index 0 means we are dividing items over the first two bins
        index = 0
        self.items.sort()

        for item_i in range(len(self.items)):
            # for uneven we add to the first bin, otherwise the second bin
            f_or_s = item_i % 2
            zero_index = np.argmin(self.x[index] != 0)
            if np.sum(self.x[index + f_or_s]) + self.items[item_i] <= self.c:
                # if it fits we add the item to the bin
                self.x[index][zero_index + f_or_s] = self.items[item_i]

            # if it doesnt fit in both we update the index
            else:
                index += 1

        # if we havent generated n items we continue the process
        if self.generated < self.n:
            self.generate_new_items()
            self.greedy2()

        # otherwise we delete all empty bins and return
        else:
            self.x = self.x[~np.all(self.x == 0, axis=1)]
            return


x = OnlineBPP(100, 10)
x.greedy2()
print(x.x)
