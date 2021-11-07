from collections import deque

from environment import Environment
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from DQN import agent
import random

plt.style.use("ggplot")

# function to paly the game manually
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
	# e.pretty_print()

	for box in e.boxes:
		states = e.get_states(box)
		for state in states:
			print(state)
			pass
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

# TODO: bug with input shapes on some training samples and optimizations
def train(replay, model, target_model, done):
	"""
	Trains model. Based on Minibatch Deep Q-Learning. 

	Takes a random sample of previous states, which are stored in a replay buffer
	and updates model
	"""
	action_idx = {"u":0, "r": 1, "d": 2, "l": 3}
	lr = 0.01
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
		Y.append([current_qs])
	Y = np.array(Y)
	X = np.array(X)
	model.fit(X, Y, batch_size=batch_size, verbose=0, shuffle=True)


# testing model
def model_test():
	action_set = {0: "u", 1: "r", 2: "d", 3:'l'}
	action_idx = {"u":0, "r": 1, "d": 2, "l": 3}
	epsilon = 1 # Epsilon-greedy algorithm, initialized to 1 so every step is random to begin with 
	max_epsilon = 1 
	min_epsilon = 0.01 # minimum always explore with 1% probability
	decay = 0.01 # rate of decay for epislon


	# instantiated environment
	file = "./input/sokoban01.txt"
	e = Environment()
	e.read_file(file)

	# get input data for DQN model
	size = e.height * e.width
	# create two models, agent and target model
	model = agent((None, size), 4)
	target_model = agent((size, ), 4)
	target_model.set_weights(model.get_weights())

	# replay buffer to store previous experiences
	replay_buffer = deque(maxlen=50000)

	# training variables
	steps_to_update_t_m = 0
	rewards = []
	epochs = 10
	for epoch in range(epochs):
		total_R = 0
		done = False
		step = 0

		# reset from file
		e.read_file(file)

		# print(f"Training: EPOCH: {epoch}, TOTAL REWARD: {total_R}")
		for i in range(200):
			# print("TOTAL REWARD: ", total_R)
			int_env = e.to_int()
			mean = int_env.mean()
			std = int_env.std()
			state = int_env.reshape( 1, e.height * e.width)
			state = state - mean/ std # standardizing
			# e.pretty_print()
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
			prev_state = deepcopy(state)
			reward, done = e.move(action_set[action])	
			replay_buffer.append([prev_state, action_set[action], reward, state, done])

			if steps_to_update_t_m % 4 == 0 or done:
				train(replay_buffer, model, target_model, done)

			total_R += reward

			if done:
				print("TOTAL TRAINING REWARDS: ", total_R)
			if steps_to_update_t_m >= 100:
				target_model.set_weights(model.get_weights())
				steps_to_update_t_m = 0
			step += 1
		if len(rewards) > 0 and (total_R > max(rewards)):
			print("BEST RESULT SO FAR: ")
			e.pretty_print()
		rewards.append(total_R)
		epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * epoch)
	plt.plot(range(1, epochs+1), rewards)
	plt.xlabel("Epoch")
	plt.ylabel("Reward")
	plt.show()



if __name__ == "__main__":
	# main()
	model_test()
