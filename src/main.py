from collections import deque

from tensorflow.python.ops.gen_batch_ops import batch
from environment import Environment
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from model2 import agent
import random

plt.style.use("ggplot")

def main():
	"""
	format of input file:
	line0 = sizeH sizeV
	line1 = number of boxes and coordinates
	line2 = number of boxes and coordinates
	line3 = number of storage locations and coordinates
	line4 = player's starting coordinates =
	"""
	file = "./input/sokoban01.txt"
	e = Environment()
	e.read_file(file)
	moveMap = {"w": "u", "d": "r", "a": "l", "s": "d"}
	move = ""
	moves = []
	# path_finder.path([0,0], [5, 5], 10, e.board)
	print(e.player)
	print(e.board)
	e.pretty_print()
	print(e.board.size * e.board.itemsize)

	for box in e.boxes:
		states = e.get_states(box)
		for state in states:
			print(state)
		print("GET STATES IN BYTES: ", state.size * state.itemsize)
	print(e.board)
	while move != "q" and not e.terminal():
		e.pretty_print()
		print(e.parseActions())
		move = input("Enter move (q to quit): ").lower()
		if move in moveMap.keys():
			moves.append(move)
			reward = e.move(moveMap[move])
			print("REWARD: ", reward)
	if e.terminal():
		e.pretty_print()
		print("You won! Here were your moves: ", moves)
	else:
		print("you quit")
def train(replay, model, target_model, done):
	action_idx = {"u":0, "r": 1, "d": 2, "l": 3}
	lr = 0.1
	gamma = 0.9
	
	MIN_REPLAY_SIZE = 1000
	if len(replay) < MIN_REPLAY_SIZE:
		return
	
	batch_size = 64 * 2
	mini_batch = random.sample(replay, batch_size)
	current_states = np.array([transition[0] for transition in mini_batch])
	current_q = model.predict(current_states)
	new_states = np.array([transition[3] for transition in mini_batch])
	future_qs = target_model.predict(new_states)

	X=[]
	Y=[]
	
	for i, (state, action, reward, new_state, done) in enumerate(mini_batch):
		if not done:
			max_future_q = reward + gamma * np.max(future_qs[i])
		else:
			max_future_q = reward
		
		current_qs = current_q[i][0]
		current_qs[action_idx[action]] = (1-lr) * current_qs[action_idx[action]] + lr * max_future_q

		X.append(state)
		Y.append(current_qs)
	model.fit(X, Y, batch_size=batch_size, verbose=0, shuffle=True)

def model_test():
	action_set = {0: "u", 1: "r", 2: "d", 3:'l'}
	action_idx = {"u":0, "r": 1, "d": 2, "l": 3}
	epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
	max_epsilon = 1 # You can't explore more than 100% of the time
	min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
	decay = 0.01
	file = "./input/sokoban01.txt"
	e = Environment()
	e.read_file(file)
	size = e.height * e.width
	model = agent((size, ), 4)
	target_model = agent((size, ), 4)
	target_model.set_weights(model.get_weights())
	replay_buffer = deque(maxlen=50000)



	steps_to_update_t_m = 0

	for epoch in range(20):
		total_R = 0
		done = False
		step = 0
		# while not done:
		# e.reset()
		for i in range(10):
			int_env = e.to_int()
			mean = int_env.mean()
			std = int_env.std()
			state = int_env.reshape( 1, e.height * e.width)
			state = state - mean/ std # standardizing
			e.pretty_print()
			steps_to_update_t_m += 1
		
			valid_actions = e.parseActions()
			valid_idx = [action_idx[i] for i in valid_actions]

			rn = np.random.rand()
			action = ""
			if rn <= epsilon:
				action = np.random.choice(valid_idx)
			else:
				predicted = model.predict(state).flatten()
				action = np.argmax(predicted)
				print(predicted)
			prev_state = deepcopy(state)
			reward, done = e.move(action_set[action])	
			replay_buffer.append([prev_state, action_set[action], reward, state, done])

			if steps_to_update_t_m % 4 == 0 or done:
				print("UPDATING")
				train(replay_buffer, model, target_model, done)

			total_R += reward

			print(f"TOTAL REWARD: {total_R}... STEP: {step} ")
			if done:
				print("TOTAL TRAINING REWARDS: ", total_R)
			if steps_to_update_t_m >= 100:
				print("UPDATING TARGET MODEL")
				target_model.set_weights(model.get_weights())
				steps_to_update_t_m = 0
			step += 1
		epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * epoch)



if __name__ == "__main__":
	main()
	# model_test()
