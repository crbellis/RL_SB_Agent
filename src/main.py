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
	file = "./input/sokoban00.txt"
	e = Environment()
	e.read_file(file)
	moveMap = {"w": "u", "d": "r", "a": "l", "s": "d"}
	move = ""
	moves = []

	# test = heuristic("m", [[box[0]-1, box[1]] for box in e.boxes], [[stor[0]-1, stor[1]-1] for stor in e.storage])
	# print(test)

	# path_finder.path([0,0], [5, 5], 10, e.board)
	# e.pretty_print()
	# for box in e.boxes:
	# 	# e.get_states(box)
	# 	print(e.parseActions(box)

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
	"""
	Trains model. Based on Minibatch Deep Q-Learning. 

	Takes a random sample of previous states, which are stored in a replay buffer
	and updates model
	"""
	action_idx = {"u":0, "r": 1, "d": 2, "l": 3}
	lr = 0.01
	gamma = 0.8
	
	MIN_REPLAY_SIZE = 1000
	if len(replay) < MIN_REPLAY_SIZE:
		return
	
	batch_size = 64 * 2
	# random batch sampling - helps speed up training
	mini_batch = random.sample(replay, batch_size)
	# get current states
	current_states = np.array([transition[0] for transition in mini_batch])
	# get current states and Q-values
	current_q = model.predict(current_states)
	# get next states and Q-values
	new_states = np.array([transition[3] for transition in mini_batch])
	future_qs = target_model.predict(new_states)

	X=[]
	Y=[]
	
	# loop through sample and calculate Q-value using Bellman Equation
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
	# fit model to Q-values
	model.fit([X], Y, batch_size=batch_size, verbose=0, shuffle=True)


# testing model
def model_test():
	"""
	Model is trained using 2 neural networks, these networks use the same 
	architecture but have different weights. We can specify when the weights 
	from the model network are transferred to the target model

	Output of the models is a 4 element array - a Q-value for each of the
	possible actions. The best action is the one with the highest Q-value
	"""
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
	model = agent((1, size), 4)
	target_model = agent((1, size), 4)
	target_model.set_weights(model.get_weights())

	# replay buffer to store previous experiences
	replay_buffer = deque(maxlen=50000)

	# training variables
	steps_to_update_t_m = 0
	rewards = []
	epochs = 10
	for epoch in range(epochs):
		moves = []
		total_R = 0
		done = False
		step = 0

		# reset from file
		e.read_file(file)

		for i in range(350):
			int_env = e.to_int()
			mean = int_env.mean()
			std = int_env.std()
			state = int_env.reshape( 1, e.height * e.width)
			state = state - mean / std # standardizing
			steps_to_update_t_m += 1
		
			# get valid actions in current state
			valid_actions = e.parseActions()
			valid_idx = [action_idx[i] for i in valid_actions]

			# choose an action
			rn = np.random.rand()
			action = ""
			if rn <= epsilon:
				action = np.random.choice(valid_idx)
			else:
				# get Q-values and action of value with highest Q-value
				predicted = model.predict([state]).flatten()
				action = np.argmax(predicted)
			prev_state = deepcopy(state)
			reward, done = e.move(action_set[action])	
			moves.append(action_set[action])

			# experience replay - used to store previous game states
			replay_buffer.append([prev_state, action_set[action], reward, state, done])

			# update the model every 4 steps
			if steps_to_update_t_m % 4 == 0 or done:
				train(replay_buffer, model, target_model, done)

			total_R += reward

			if done:
				# print("TOTAL TRAINING REWARDS: ", total_R)
				print(moves)
				break
			
			# update target model weights every 100 steps
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
