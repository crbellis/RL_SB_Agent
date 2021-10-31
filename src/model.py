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