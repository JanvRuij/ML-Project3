import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from OBPP_Env import BalanceEnv


class OnlineBPP:
    def __init__(self, n, m) -> None:
        # number of items
        self.n = n
        # number of items after which the new items are generated
        self.m = m
        # generate m random items of maximum size 100
        self.items = np.random.randint(1, 100, (m, 1))
        # capacity of the bins
        self.c = 200
        # solution vector
        self.x = np.zeros((n, n))
        # to keep track of items generated
        self.generated = m

    def generate_new_items(self):
        self.items = np.round(np.random.rand(self.m)*100, 0)
        self.generated += self.m

    # keep adding until it doesnt fit anymore (sorted first)
    def greedy1(self):
        self.items.sort()
        for item in self.items:
            # sort the indexes from high row sum to low
            sorted_indices = np.flip(np.argsort(self.x.sum(axis=1), axis=0))
            # we want to append the item to the bin with the most items
            for idx in sorted_indices:
                # check if the item fits
                if np.sum(self.x[idx]) + item <= self.c:
                    # if it fits we append it to the first open position
                    zero_index = np.argwhere(self.x[idx] == 0)[0]
                    self.x[idx][zero_index] = item
                    break

        # if we havent generated n items we continue the process
        if self.generated < self.n:
            self.generate_new_items()
            self.greedy1()

        # otherwise we delete all empty bins and return (for readibility)
        else:
            self.x = self.x[:, np.sum(self.x, axis=0) != 0]
            self.x = self.x[np.sum(self.x, axis=1) != 0]
            return

    # divide items evenly over two bins (also sorted)
    def greedy2(self):
        # index 0 means we are dividing items over the first two bins
        index = 0
        self.items.sort()
        # we try to fit item 0 first
        item_i = 0
        while item_i < len(self.items):
            # for uneven we add to the first bin, otherwise the second bin
            f_or_s = item_i % 2
            zero_index = np.argmin(self.x[index] != 0)
            if np.sum(self.x[index + f_or_s]) + self.items[item_i] <= self.c:
                # if it fits we add the item to the bin
                self.x[index][zero_index + f_or_s] = self.items[item_i]
                # go to next item
                item_i += 1

            # if it doesnt fit in both we update the index
            else:
                index += 1
        # if we havent generated n items we continue the process
        if self.generated < self.n:
            self.generate_new_items()
            self.greedy2()

        # otherwise we delete all empty bins and return (for readibility)
        else:
            self.x = self.x[:, np.sum(self.x, axis=0) != 0]
            self.x = self.x[np.sum(self.x, axis=1) != 0]
            return


# Setting up the environment
env = BalanceEnv()

# Building the nnet that approximates q
# number of items visible = number of actions
n_actions = env.m
print(n_actions)
input_dim = env.observation_space.shape[0]
model = Sequential()
model.add(Dense(250, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(n_actions, activation='softmax'))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

modelT = Sequential()
modelT.add(Dense(250, activation='sigmoid'))
modelT.add(Dense(50, activation='sigmoid'))
modelT.add(Dense(n_actions, activation='softmax'))
modelT.compile(optimizer=Adam(learning_rate=0.001), loss='mse')


# Experience replay
def replay(replay_memory, minibatch_size):
    minibatch = np.random.choice(replay_memory, minibatch_size, replace=False)
    s_l = np.array(list(map(lambda x: x['s'], minibatch)))
    a_l = np.array(list(map(lambda x: x['a'], minibatch)))
    r_l = np.array(list(map(lambda x: x['r'], minibatch)))
    sprime_l = np.array(list(map(lambda x: x['sprime'], minibatch)))
    done_l = np.array(list(map(lambda x: x['done'], minibatch)))
    qvals_sprime_l = modelT.predict(sprime_l, verbose="0")  # Predict q-values for all actions in s' with target nn
    target_f = model.predict(s_l, verbose="0") # Predict q-values for all actions in s with nn
    # q-values update
    for i, (s, a, r, qvals_sprime, done) in enumerate(zip(s_l, a_l, r_l, qvals_sprime_l, done_l)):
        if done:
            target_f[i][a] = 0
        else:    
            target_f[i][a] = r + gamma * max(qvals_sprime) # Update q-value of action a in state s by replacing the prediction with the observed q-value 
    history = model.fit(s_l,target_f, epochs=1, verbose=0, batch_size=minibatch_size) # Train the nn
    avgloss.append(history.history["loss"][0])
    return model


n_episodes = 2000
it = 0
gamma = 0.993
epsilon = 1.0
epsilon_multiplier = 0.93
epsilon_min = 0.0001
minibatch_size = 578
memory_minsize = 1086
n_update = 8
r_sums = [0]
g_sums = [0]
r_betters = [0]
loss = []
avgloss = []
replay_memory = []  # replay memory holds s, a, r, s'
mem_max_size = 10000

for n in range(n_episodes):
    s = env.reset()
    done = False
    r_sum = 0.0
    while not done:
        it += 1
        # Predict q-values for all actions
        qvals_s = model.predict(s.reshape(1, input_dim), verbose="0")
        a = 0
        # Choose action to be epsilon-greedy
        if np.random.random() < epsilon:
                a = env.action_space.sample()
        else:
            a = np.argmax(qvals_s)
        sprime, r, done, info = env.step(a)
        r_sum += r
        # add to memory, respecting memory buffer limit
        if len(replay_memory) > mem_max_size:
            replay_memory.pop(0)
        replay_memory.append({"s": s, "a": a, "r": r, "sprime": sprime, "done": done})
        # Update state
        s = sprime
        # Train the nnet that approximates q(s,a), using the replay memory
        if len(replay_memory) > memory_minsize:
            
            epsilon *= epsilon_multiplier
            model = replay(replay_memory, minibatch_size)
            if it % n_update == 0:
                modelT.set_weights(model.get_weights())

    if n % 10 == 0:
        print("######", n, r_sums[-1], g_sums[-1], r_sums[-1] - g_sums[-1], r_betters[-1], "######")
        r_sums.append(0)
        g_sums.append(0)
        r_betters.append(0)
        if len(avgloss) > 0:
            loss.append(sum(avgloss)/len(avgloss))
            avgloss = []
    # env.render()
    r_sums[-1] += r_sum
    g_sums[-1] += env.greedy
    if r_sum > env.greedy:
        r_betters[-1] += 1
    elif r_sum < env.greedy:
        r_betters[-1] -= 1

model.save("model_OBPP")
d_sums = [g_sums[i] - r_sums[i] for i in range(len(g_sums))]
plt.plot(d_sums, 'ro', markersize=1)
plt.show()
plt.clf
plt.plot(r_betters, 'ro', markersize=1)
plt.show()
plt.clf
plt.plot(loss, 'ro', markersize=1)
plt.show()
print(r_betters)
