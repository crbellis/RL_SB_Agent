import tensorflow as tf
import numpy as np
from collections import deque

replay_buffer = deque(maxlen=2000)
input_shape = (1, 64)
model  = tf.keras.models.Sequential([
	tf.keras.layers.Dense(64, activation='relu', input_shape=(64,)),
	tf.keras.layers.Dense(32, activation="relu"),
	tf.keras.layers.Dense(16, activation="relu"),
	tf.keras.layers.Dense(4)
])

model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=1e-3))

def epsilon_greedy_policy(state, epislon = 0):
	if np.random.rand() < epislon:
		return np.random.randint(4)
	else:
		Q_values = model.predict(state)
		return np.argmax(Q_values) 

def play_one_move(env, state, epsilon):

	action_set = {
		0: 'u',
		1: 'd',
		2: 'l',
		3: 'r',
	}
	action = epsilon_greedy_policy(state, epsilon)
	reward, done = env.move(action_set[action])
	print(action_set[action])
	return reward, done

def sample_experiences(batch_size):
	indices = np.random.randint(len(replay_buffer), size=batch_size)
	batch = [replay_buffer[index] for index in indices]
	rewards, done = [np.array([experience[field_index] for experience in batch])
		for field_index in range(5)
	]
	return rewards, done


def train_test(batch_size):
	experiences = sample_experiences(batch_size=batch_size)
	print(experiences)


	# rewards = []
	# batch_size = 32
	# optimizer=tf.keras.optimizers.SGD(lr=0.1)
	# loss_fn = tf.keras.losses.mean_squared_error
	# model.compile(loss=loss_fn, optimizer=optimizer)

	# input_shape = e.height * e.width
	# gamma = 0.85
	# epislon = 0.1
	# action_set = {0: "u", 1: "r", 2: "d", 3:'l'}
	# action_idx = {"u":0, "r": 1, "d": 2, "l": 3}
	# # print(e.to_numpy().reshape(1, input_shape))
	# epoch = 0
	# while not e.terminal() and epoch < 10000:
	# 	# e.pretty_print()
	# 	state_1 = (e.to_numpy().reshape(1, input_shape) 
	# 		+ np.random.rand(1, input_shape) / 100.0)

	# 	valid_actions = e.parseActions()
	# 	valid_idx = [action_idx[i] for i in valid_actions]
	# 	q_vals = model.predict(state_1)
	# 	for val in q_vals[0]:
	# 		idx = np.where(q_vals[0] == val)[0][0]
	# 		# print(idx, val)
	# 		if idx not in valid_idx:
	# 			q_vals[0][idx] = float('-inf')

	# 	sample_ep = np.random.rand()
	# 	# print(q_vals, valid_actions, valid_idx)
	# 	if sample_ep <= epislon:
	# 		print("RANDOM ACTION")
	# 		action = np.random.choice(valid_idx)
	# 	else:
	# 		action = np.argmax(q_vals[0])
	# 	action_char = action_set[action]
	# 	# print(action_char)
	# 	reward = e.move(action_char)
	# 	rewards.append(reward[0])
	# 	q_val = q_vals[0][action]

	# 	state_2 = (e.to_numpy().reshape(1,input_shape)  
	# 		+ np.random.rand(1, input_shape) / 100.0)


	# 	# print(f"EPOCH: {epoch+1}\nTotal reward: {e.totalReward}.\n")
	# 	# e.pretty_print()
	# 	if e.terminal():
	# 		next_q = 0
	# 		print("YOU WON")
	# 		break
			
	# 	else:
	# 		next_q = tf.stop_gradient(model.predict(state_2))
	# 		next_action = np.argmax(next_q[0])
	# 		next_q_val = next_q[0, next_action]

	# 	observed_q = reward + (gamma*next_q_val)
	# 	loss_val = loss_fn(observed_q, q_val)
	# 	# optimizer.apply_gradients(zip(grads, model.trainable_variables))

	# 	state_1 = state_2
	# 	epoch += 1
	# print(rewards)
	# plt.plot(range(1, epoch+1), rewards, label="rewards over time")
	# plt.xlabel("Epoch")
	# plt.ylabel("Rewards")
	# plt.legend()
	# plt.show()