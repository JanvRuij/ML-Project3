from OBPP_Env import BalanceEnv
import tensorflow as tf
import numpy as np
import pickle


env = BalanceEnv()
model = tf.keras.models.load_model("model_OBPP")
input_dim = env.observation_space.shape[0]
output_list = []
greedy_win = 0
neural_win = 0
# lets race the two models
for i in range(500):
    s = env.reset()
    # we store the size of the instance
    i_s = env.nbItems
    # we store the greedy value
    g_v = env.greedy
    # keep iterating untill we are done
    done = False
    # have to store statistics
    greedy_win = 0
    neural_win = 0
    neural_reward = 0
    while not done:
        qvals_s = model.predict(s.reshape(1, input_dim), verbose="0")
        a = np.argmax(qvals_s)
        sprime, r, done, info = env.step(a)
        s = sprime

    neural_reward = env.total_reward
    print(neural_reward)
    print(g_v)
    if neural_reward > g_v:
        print("Neural wins!")
        neural_win += 1
    elif neural_reward < g_v:
        print("Greedy wins!")
        greedy_win += 1

    output_list.append([i_s, g_v, neural_reward])


# Writing the list to a file using pickle
with open('list.pkl', 'wb') as file:
    pickle.dump(output_list, file)

print(greedy_win)
print(neural_win)
