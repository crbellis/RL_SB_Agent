import tensorflow as tf
import numpy as np
from collections import deque
import random

def agent(state_shape, action_shape):
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    """
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(tf.keras.layers.Dense(12, activation='relu', kernel_initializer=init))
    model.add(tf.keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
    return model

def get_qs(model, state):
    return model.predict(state.reshape([1, state.shape[0]]))[0]

def train(replay_memory, model, target_model, done):
    learning_rate = 0.001 # Learning rate
    discount_factor = 0.9

    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 64 * 2
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(observation)
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)