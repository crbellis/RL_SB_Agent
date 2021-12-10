from collections import deque
import time
from os import listdir
from os.path import isfile, join
from environment import Environment
import numpy as np
from copy import deepcopy
from DQN import agent
import tensorflow as tf
import random


def train(replay, model, target_model, done, size):
	"""
	Trains model. Based on Minibatch Deep Q-Learning. 

	Takes a random sample of previous states, which are stored in a replay buffer
	and updates model
	"""
	action_idx = {"u":0, "r": 1, "d": 2, "l": 3}
	gamma = 0.9 # how do rewards in the future affect Q-values
	MIN_REPLAY_SIZE = 1024
	if len(replay) < MIN_REPLAY_SIZE:
		return

	batch_size = 512

	# random batch sampling - helps speed up training
	mini_batch = random.sample(replay, batch_size)
	# get current states
	current_states = np.array([transition[0] for transition in mini_batch])
	# get current states and Q-values
	target = model.predict(current_states.reshape(batch_size, 1, size))
	# get next states and Q-values
	next_states = np.array([transition[3] for transition in mini_batch])
	next_qs = target_model.predict(next_states.reshape(batch_size, 1, size))
	q_next = tf.math.reduce_max(next_qs, axis=1, keepdims=True).numpy()

	q_target=np.copy(target)
	# reshaping for fitting to model
	q_target = q_target.reshape(batch_size, 4)
	q_next = q_next.reshape(batch_size, 4)
	# loop through sample and calculate Q-value using Bellman Equation
	for i, (state, action, reward, new_state, done) in enumerate(mini_batch):
		# if the state is terminal we use the reward value to train, otherwise Bellman Equation is used to update
		if done:
			q_target[i, action_idx[action]] = reward
		else:
			q_target[i, action_idx[action]] = reward + gamma * q_next[i, action_idx[action]]

	history = model.train_on_batch(
		current_states.reshape(batch_size, 1, size), 
		q_target.reshape(batch_size, 1, 4), 
		return_dict=True
	)
	return history

def create_agent(file: str, game_epochs: int, moveLimit: int):
	"""
	Model is trained using 2 neural networks, these networks use the same 
	architecture but have different weights. We can specify when the weights 
	from the model network are transferred to the target model

	Output of the models is a 4 element array - a Q-value for each of the
	possible actions. The best action is the one with the highest Q-value

	Returns
	-----
	tuple -> list of moves made to solve the game, time in minutes it took to solve
	"""
	# for timer
	start = time.time()
	# dictionary of action sets to map movements in indicies in arrays
	action_set = {0: "u", 1: "r", 2: "d", 3:'l'}
	# inverse of the above
	action_idx = {"u":0, "r": 1, "d": 2, "l": 3}

	epsilon = 1 # Epsilon-greedy algorithm, initialized to 1 so every step is random to begin with 
	max_epsilon = 1
	min_epsilon = 0.1 # minimum always explore with 1% probability
	decay = 0.01 # rate of decay for epislon

	# instantiated environment
	e = Environment()
	# get input data for DQN model
	e.read_file(file)
	e.getOrder2()
	size = e.height * e.width
	# create two models, agent and target model
	model = agent((None, size), 4)
	target_model = agent((None, size), 4)
	# copy weights to target model from main model
	target_model.set_weights(model.get_weights())

	# replay buffer to store previous experiences
	replay_buffer = deque(maxlen=50000)

	# training variables for analysis
	rewards = []
	steps_to_update_t_m = 1
	epochs = 0
	acc = []
	avg_acc = []
	loss = []
	avg_loss = []
	isTerminal = False
	history=None

	# loop for number of epochs, end when the game has been solved. 1 Epoch is 1 game
	while(not isTerminal and epochs < game_epochs):
		# increment epochs
		epochs += 1
		# keep track of all moves
		moves = []
		# total reward
		total_R = 0
		# variable for game state, done or not
		done = False
		# number of moves the agent made
		step = 0

		# reset from file
		e.read_file(file)

		# getting the float version of the board
		state = e.to_float()
		# min max scaling for feeding to neural net
		state /= 6

		# list of repeat moves
		repeats = []
		# number of block movements and bonusMoves rewarded for pushing blocks
		blockMovements = 0
		bonusMoves = 0

		# game loop: loop until the moveLimit has been met or the game is done (deadlock or solved)
		while(not done and blockMovements < moveLimit + bonusMoves):
			
			# if player is not by a box, go to random box
			if (not ([e.player[0], e.player[1] - 1] in e.boxes 
				or [e.player[0], e.player[1] + 1] in e.boxes 
				or [e.player[0]-1, e.player[1]] in e.boxes 
				or [e.player[0] + 1, e.player[1]] in e.boxes)):
				# gets possilbe block moves (finds path to coordinates)
				block_moves = e.get_moves()
				# if there is an available move
				if len(block_moves) > 0:
					# select random block
					block_moves = random.sample(block_moves, 1)
					# loop through action in block move i.e. [2, 2], "u"
					for block, action in block_moves:
						# get the inverse coordinate
						inverse = {"u":"d", "l":"r", "r":"l", "d":"u"}
						dest = block.copy()
						"""
						get the otherside of the block for the desired movement, 
						i.e. push r requires the left side of the boxe to be accessible
						"""
						if action in ("u", "d"):
							dest[1] += e.movements[inverse[action]]
						else:
							dest[0] += e.movements[inverse[action]]
						
						# will not push blocks in the process of moving to
						moves_to_block = e.go_to(dest) # returns list of moves to destination
						# append all moves taken to get to new coordinate to moves list
						[moves.append(move) for move in moves_to_block]
						# add negative penalty for every move
						total_R += len(moves_to_block) * -1
						# add len of moves to number of steps
						step += len(moves_to_block)
						# update state
						state = e.to_float()
						state /= 6
				else:
					# otherwise no more movements, end game
					break
			# get all possible actions from agent location, u, d, l, r
			valid_actions = e.parseActions()
			# get the index of these actions
			valid_idx = [action_idx[i] for i in valid_actions]

			# choose an action
			rn = np.random.rand()
			action = ""
			predicted = []
			# if random number is less than epsilon choose a random action from valid actions
			if rn <= epsilon:
				action = np.random.choice(valid_idx)

			# else predict the current Q-values
			else:
				# get Q-values and action of value with highest Q-value
				predicted = model.predict(state.reshape(1, 1, size)).flatten()
				# remove invalid actions
				for i in range(4):
					if i not in valid_idx:
						predicted[i] = float("-inf")

				# select action with highest Q-value
				action = np.argmax(predicted)
			repeats.append(action)
			# check for repeated actions, redo the above selection if repeat
			if repeats[:3] == repeats[2:5] or repeats[-4:-2] == repeats[-2:]:
				if len(predicted) == 0:
					action = np.random.choice(valid_idx)	
				else:
					predicted[action] = float('-inf')
					action = np.argmax(predicted)
				repeats = repeats[2:]

			if len(repeats) > 4:
				repeats.pop(0)

			# copy previos state and make the select action
			prev_state = deepcopy(state)
			reward, done = e.move(action_set[action])

			if e.movedBox:
				blockMovements += 1

			# if high positive reward allow agent to make more moves
			if reward > 20:
				bonusMoves += int(reward / (e.height + e.width))
			
			# new state
			state = deepcopy(e.to_float())
			state /= 6

			# append actions to moves
			moves.append(action_set[action])

			# experience replay - storing previous states, actions, rewards, states, and terminality
			replay_buffer.append([prev_state, action_set[action], reward, state, done])

			# every 2 steps train the model
			if step % 2 == 0:
				history = train(replay_buffer, model, target_model, done, size)
				# for checking model accuracy over time
				if history:
					acc.append(history["accuracy"])
					loss.append(history["loss"])
					# update steps to update target model
					steps_to_update_t_m += 1

			# update target model after 1000 trainings
			if steps_to_update_t_m % 1000 == 0 and len(replay_buffer) > 1024:
				# update target model weights every 100 steps
				target_model.set_weights(model.get_weights())
				print("Copying main model weights to target model")
				steps_to_update_t_m = 1

			# update total reward and increment step
			total_R += reward
			step += 1

			# if game done break out of the loop
			if done:
				break

		# counts for various metrics of interest, box count, accuracy, loss, etc
		if history:
			acc.append(history["accuracy"])
			loss.append(history["loss"])

		boxCount = 0
		for box in e.boxes:
			if box in e.storage:
				boxCount += 1
		print(f"{epochs}: reward: {total_R:.0f}. (epsilon: {epsilon:.2f}).", 
			f" No. of boxes in storage: {boxCount}/{len(e.boxes)}.",
			f" No. of moves: {len(moves)}",
			f" Minutes elapsed: {(time.time() -  start) / 60:.2f}",
			f"Length of replay: {len(replay_buffer)}")

		epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * epochs) # change to epoch 

		rewards.append(total_R)
		if len(loss)>0:
			avg_loss.append(sum(loss)/len(loss))
			avg_acc.append(sum(acc)/len(acc))

		# prints board state if terminal# prints board state if terminal# prints board state if terminal
		if e.terminal():
			print("Level solved. Number of moves: ", len(moves))
			e.pretty_print()
			break
			
	end = time.time()

	# returns lists of moves and time to execute in minutes
	return [move.upper() for move in moves], (end-start)/60

def test_model(tests = list):
	"""
	tests - list of input file numbers
	"""
	test_times  = {}
	# loops through tests passed prints out len of current moves, along with all the moves
	for test in tests: 
		print("CURRENT FILE: ", test)
		moves, time_ = create_agent(test, game_epochs=1000, moveLimit=300)
		print(len(moves), " ".join(moves))
		print(f"FILE: {test}. Time to run (in minutes): {round(time_, 4)}")
		test_times[test] = {"time": round(time_, 4), "moves": moves}

	print(test_times)

if __name__ == "__main__":
	"""
	runs the model on all files. 
	test_model() accepts a list of files to test, if you wish to test one 
	file simple pass ["path_to_test.txt"] as the argument
	"""
	# finds all the sokoban.txt files in benchmarks folders
	files = ["./benchmarks/"+f for f in listdir("./benchmarks/") if isfile(join("./benchmarks/", f)) and "sokoban" in f]
	files.sort()
	test_model(files)