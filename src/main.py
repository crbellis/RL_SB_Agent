from collections import deque
import time
from os import listdir
from os.path import isfile, join

from tensorflow.python.ops.gen_batch_ops import batch
from environment import Environment
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from DQN import agent
import tensorflow as tf
from train import train as train2
# print(tf.config.list_physical_devices())
# tf.config.run_functions_eagerly(False)
# from tensorflow.python.compiler.mlcompute import mlcompute
# mlcompute.set_mlc_device(device_name='gpu')
import random
# from path_finder import heuristic
from storageOrder import getOrder
plt.style.use("ggplot")



# function to play the game manually
def play():
	"""
	format of input file:
	line0 = sizeH sizeV
	line1 = number of boxes and coordinates
	line2 = number of boxes and coordinates
	line3 = number of storage locations and coordinates
	line4 = player's starting coordinates =
	"""
	file = "./input_files/sokoban01.txt"

	# deadSpaces = [[1,1], [1,2], [1, 3], [1, 4]]
	e = Environment()
	e.read_file(file)
	# e.deadSpaces = deadSpaces
	moveMap = {"w": "u", "d": "r", "a": "l", "s": "d"}
	move = ""
	moves = []

	size = e.height * e.width
	# create two models, agent and target model
	model = agent((None, size), 4)
	target_model = agent((None, size), 4)
	target_model.set_weights(model.get_weights())

	# replay buffer to store previous experiences
	replay_buffer = deque(maxlen=50000)

	# training variables
	steps_to_update_t_m = 0
	rewards = []
	epochs = 100
	acc = []
	avg_acc = []
	loss = []
	avg_loss = []
	# test = heuristic("m", [[box[0]-1, box[1]] for box in e.boxes], [[stor[0]-1, stor[1]-1] for stor in e.storage])
	# print(test)

	# path_finder.path([0,0], [5, 5], 10, e.board)
	# e.pretty_print()
	# for box in e.boxes:
	# 	# e.get_states(box)
	# 	print(e.parseActions(box)

	prevCoords = []
	state = e.to_float()
	state /= 6
	while move != "q" and not e.terminal():
		steps = 0
		e.pretty_print()
		# print(e.to_float())

		print(e.get_moves())
		action_set = {0: "u", 1: "r", 2: "d", 3:'l'}
		action_idx = {"u":0, "r": 1, "d": 2, "l": 3}
		pred = model.predict(state.reshape(1, 1, size)).flatten()
		valid_actions = e.parseActions()
		print(valid_actions)
		valid_idx = [action_idx[i] for i in valid_actions]
		# for i in range(3):
		# 	if i in valid_idx:
		# 		pred[i] = valid_idx[i]
		# 	else:
		# 		pred[i] = float("-inf")
		print(valid_idx)
		for i in range(4):
			if i not in valid_idx:
				pred[i] = float("-inf")	
		print(action_set[np.argmax(pred)])
		print(pred)
		move = input("Enter move (q to quit): ").lower()
		if move in moveMap.keys():
			prevCoords = e.player
			moves.append(move)


			prev_state = deepcopy(state)
			reward, done = e.move(moveMap[move])
			print(reward)
			state = e.to_float()
			state /= 6
			replay_buffer.append([prev_state, moveMap[move], reward, state, done])
			if done:
				break

			if steps % 4 == 0 or done:
				print("Training")
				history = train(replay_buffer, model, target_model, done, size)
			
			if steps_to_update_t_m >= 100:
				target_model.set_weights(model.get_weights())
				steps_to_update_t_m = 0

			steps += 1

		elif move == "z":
			# [x, y]
			e.undo([prevCoords, moveMap[moves[-1]]])
	if e.terminal():
		e.pretty_print()
		print("You won! Here were your moves: ", moves)
	else:
		print("you quit")

#inspect the model after training
def inspect(model, target_model, file):
	e = Environment()
	e.read_file(file)
	moveMap = {"w": "u", "d": "r", "a": "l", "s": "d"}
	move = ""
	moves = []

	size = e.height * e.width

	# replay buffer to store previous experiences
	replay_buffer = deque(maxlen=50000)

	# training variables
	steps_to_update_t_m = 0

	prevCoords = []
	print(e.get_moves())
	state = e.to_float()
	state /= 6
	state += np.random.rand(e.height, e.width) / 10 # adding noise
	while move != "q" and not e.terminal():
		steps = 0
		print(e.to_float())
		print(e.parseActions())


		action_set = {0: "u", 1: "r", 2: "d", 3:'l'}
		action_idx = {"u":0, "r": 1, "d": 2, "l": 3}
		pred = model.predict(state.reshape(1, 1, size)).flatten()
		valid_actions = e.parseActions()
		valid_idx = [action_idx[i] for i in valid_actions]
		# for i in range(3):
		# 	if i in valid_idx:
		# 		pred[i] = valid_idx[i]
		# 	else:
		# 		pred[i] = float("-inf")
		print(pred, [pred[i] for i in valid_idx])
		for i in range(4):
			if i not in valid_idx:
				pred[i] = float("-inf")
		print(action_set[np.argmax(pred)])
		move = input("Enter move (q to quit): ").lower()
		if move in moveMap.keys():
			prevCoords = e.player
			moves.append(move)


			prev_state = deepcopy(state)
			reward, done = e.move(moveMap[move])
			print(reward)
			state = e.to_float()
			state /= 6
			state += np.random.rand(e.height, e.width) / 10 # adding noise


			replay_buffer.append([prev_state, moveMap[move], reward, state, done])
			if done:
				break

			if steps % 4 == 0 or done:
				print("Training")
				history = train(replay_buffer, model, target_model, done, size)
			
			if steps_to_update_t_m >= 100:
				target_model.set_weights(model.get_weights())
				steps_to_update_t_m = 0

			steps += 1

		elif move == "z":
			# [x, y]
			e.undo([prevCoords, moveMap[moves[-1]]])
	if e.terminal():
		e.pretty_print()
		print("You won! Here were your moves: ", moves)
	else:
		print("you quit")


def train(replay, model, target_model, done, size):
	"""
	Trains model. Based on Minibatch Deep Q-Learning. 

	Takes a random sample of previous states, which are stored in a replay buffer
	and updates model
	"""
	action_idx = {"u":0, "r": 1, "d": 2, "l": 3}
	# lr = 1e-5
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

	# X=[]
	# Y=[]
	
	q_target=np.copy(target)
	q_target = q_target.reshape(batch_size, 4)
	q_next = q_next.reshape(batch_size, 4)
	# loop through sample and calculate Q-value using Bellman Equation
	for i, (state, action, reward, new_state, done) in enumerate(mini_batch):
		if done:
			q_target[i, action_idx[action]] = reward
		else:
			q_target[i, action_idx[action]] = reward + gamma * q_next[i, action_idx[action]]
		# if not done:
		# 	expected_state_action_value = reward + (gamma * np.max(future_qs[i][0]))
		# else:
		# 	# expected_state_action_value = reward 
		
		# target = np.copy(current_q[i][0])
		# print("EXPECTED: ", expected_state_action_value, end="\t")
		# current_qs[action_idx[action]] = ((1-lr) * current_qs[action_idx[action]]) + (lr * expected_state_action_value)
		# current_qs[action_idx[action]] = lr * (expected_state_action_value - current_qs[action_idx[action]])
		# target[action_idx[action]] = reward + gamma * np.max(future_qs[i][0]) - target[action_idx[action]]
		# if reward > 0:
		# 	print("CURRENT STATE: \n", state * 6)
		# 	print("ACTION: ", action)
		# 	print("REWARD: ", reward)
		# 	print("NEW STATE: \n", new_state*6)
		# 	print("Q: ", current_qs[action_idx[action]])
		# 	print()

		# X.append(state.reshape(1, size))
		# Y.append(target.reshape(1, 4))
	
	# Y = np.array(Y)
	# X = np.array(X)
	# fit model to Q-values
	history = model.train_on_batch(
		current_states.reshape(batch_size, 1, size), 
		q_target.reshape(batch_size, 1, 4), 
		return_dict=True
	)
	return history
# testing model
def create_agent(file: str, game_epochs: int, moveLimit: int):
	"""
	Model is trained using 2 neural networks, these networks use the same 
	architecture but have different weights. We can specify when the weights 
	from the model network are transferred to the target model

	Output of the models is a 4 element array - a Q-value for each of the
	possible actions. The best action is the one with the highest Q-value
	"""
	try:
		# deadSpaces = [[1,1], [1,2], [1, 3], [1, 4]]
		start = time.time()
		action_set = {0: "u", 1: "r", 2: "d", 3:'l'}
		action_idx = {"u":0, "r": 1, "d": 2, "l": 3}

		epsilon = 1 # Epsilon-greedy algorithm, initialized to 1 so every step is random to begin with 
		max_epsilon = 1
		min_epsilon = 0.1 # minimum always explore with 1% probability
		decay = 0.01 # rate of decay for epislon

		# instantiated environment
		e = Environment()
		# get input data for DQN model
		e.read_file(file)
		e.pretty_print()
		order = e.getOrder()
		# storageOrder = getOrder(e)
		# print("ORDER TO SOLVE: ", storageOrder)
		size = e.height * e.width
		# create two models, agent and target model
		model = agent((None, size), 4)
		target_model = agent((None, size), 4)
		target_model.set_weights(model.get_weights())

		# replay buffer to store previous experiences
		replay_buffer = deque(maxlen=50000)

		# training variables
		rewards = []
		steps_to_update_t_m = 1
		epochs = 0
		acc = []
		avg_acc = []
		loss = []
		avg_loss = []
		isTerminal = False
		history=None
		# while(False):
		while(not isTerminal and epochs < game_epochs):
			epochs += 1
			moves = []
			total_R = 0
			done = False
			step = 0
			# reset from file
			e.read_file(file)
			e.setOrder(order)
			state = e.to_float()
			state /= 6

			repeats = []
			bonusMoves = 0
			while(not done and len(moves) < moveLimit + bonusMoves):
				# after Q based action
				# e.pretty_print()

				# print(len(moves))
				if (not ([e.player[0], e.player[1] - 1] in e.boxes 
					or [e.player[0], e.player[1] + 1] in e.boxes 
					or [e.player[0]-1, e.player[1]] in e.boxes 
					or [e.player[0] + 1, e.player[1]] in e.boxes)):
					# print("NOT NEXT TO BOX")
					block_moves = e.get_moves()
					# block_moves = list(zip(block_moves[::2], block_moves[1::2]))
					if len(block_moves) > 0:
						block_moves = random.sample(block_moves, 1)
						for block, action in block_moves:
							inverse = {"u":"d", "l":"r", "r":"l", "d":"u"}
							newBlock = block.copy()
							if action in ("u", "d"):
								newBlock[1] += e.movements[inverse[action]]
							else:
								newBlock[0] += e.movements[inverse[action]]
							moves_to_block = e.go_to(newBlock)
							# print("MOVES TO BLOCK: ", moves_to_block)
							[moves.append(move) for move in moves_to_block]
							total_R += len(moves_to_block) * -1
							step += len(moves_to_block)
							state = e.to_float()
							state /= 6
				# after block movement
						# e.pretty_print()
				# get valid actions in current state
				valid_actions = e.parseActions()
				# print("VALID ACTIONS: ", valid_actions)
				valid_idx = [action_idx[i] for i in valid_actions]

				# choose an action
				rn = np.random.rand()
				action = ""
				predicted = []
				if rn <= epsilon:
					action = np.random.choice(valid_idx)
				else:
					# get Q-values and action of value with highest Q-value
					predicted = model.predict(state.reshape(1, 1, size)).flatten()
					for i in range(4):
						if i not in valid_idx:
							predicted[i] = float("-inf")

					action = np.argmax(predicted)

				repeats.append(action)
				if repeats[:3] == repeats[2:5] or repeats[-4:-2] == repeats[-2:]:
					if len(predicted) == 0:
						action = np.random.choice(valid_idx)	
					else:
						predicted[action] = float('-inf')
						action = np.argmax(predicted)
					repeats = repeats[2:]

				if len(repeats) > 4:
					repeats.pop(0)

				prev_state = deepcopy(state)
				reward, done = e.move(action_set[action])	
				# print("ACTION MADE: ", action_set[action])
				# new state
				if reward > 20:
					bonusMoves += int(reward * 1/3)
				state = deepcopy(e.to_float())
				state /= 6
				# state += np.random.rand(e.height, e.width) / 1000 # adding noise

				moves.append(action_set[action])

				# experience replay - used to store previous game states
				replay_buffer.append([prev_state, action_set[action], reward, state, done])
				if step % 2 == 0:
					history = train(replay_buffer, model, target_model, done, size)
					if history:
						acc.append(history["accuracy"])
						loss.append(history["loss"])
						steps_to_update_t_m += 1

				if steps_to_update_t_m % 1000 == 0 and len(replay_buffer) > 500:
					# update target model weights every 100 steps
					target_model.set_weights(model.get_weights())
					print("Copying main model weights to target model")
					steps_to_update_t_m = 1

				total_R += reward
				step += 1
				if done:
					break

			if history:
				acc.append(history["accuracy"])
				loss.append(history["loss"])

			boxCount = 0
			for box in e.boxes:
				if box in e.storage:
					boxCount += 1
			print(file)
			print(f"{epochs}: reward: {total_R:.0f}. (epsilon: {epsilon:.2f}).", 
				f" No. of boxes in storage: {boxCount}/{len(e.boxes)}.",
				f" No. of moves: {len(moves)}",
				f" Minutes elapsed: {(time.time() -  start) / 60:.2f}",
				f"Length of replay: {len(replay_buffer)}")

			acc_val = sum(acc)/max(len(acc), 1)
			if acc_val < 90:
				epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * epochs) # change to epoch 
			else:
				epsilon = 0.2

			if len(rewards) > 0 and (total_R > max(rewards)):
				# print(f"REWARD: {total_R:.0f}\nBEST RESULT SO FAR: ")
				e.pretty_print()

			rewards.append(total_R)
			if len(loss)>0:
				avg_loss.append(sum(loss)/len(loss))
				avg_acc.append(sum(acc)/len(acc))

			if e.terminal():
				print("Level solved. Number of moves: ", len(moves))
				e.pretty_print()
				print(e.to_int())
				break
			
	except KeyboardInterrupt:
		print("interupted")

	end = time.time()

	# if len(rewards) < epochs:
	# 	for i in range(len(rewards), epochs):
	# 		rewards.append(0)

	# # print(f"FILE: {file}. MINUTES TO RUN: {(end - start)/60}")
	# plt.plot(range(1, epochs+1), rewards)
	# plt.xlabel("Epoch")
	# plt.ylabel("Reward")
	# plt.show()
	# plt.clf()
	# if len(avg_loss) > 0:
	# 	plt.plot(range(1, len(avg_loss)+1), avg_loss)
	# 	plt.xlabel("Epoch")
	# 	plt.ylabel("Average Loss")
	# 	plt.title("Avg. Loss per Epoch")
	# 	plt.show()
	# 	plt.clf()

	# 	plt.plot(range(1, len(avg_acc)+1), avg_acc)
	# 	plt.xlabel("Epoch")
	# 	plt.ylabel("Average Accuracy")
	# 	plt.show()
	# 	plt.clf()


	# inspect(model, target_model, file)	
	return [move.upper() for move in moves], (end-start)/60

def test_model(tests = list):
	"""
	tests - list of input file numbers
	"""
	test_times  = {}
	for test in tests: 
		print("CURRENT FILE: ", test)
		moves, time_ = create_agent(test, game_epochs=1000, moveLimit=200)
		print(len(moves), " ".join(moves))
		print(f"FILE: {test}. Time to run (in minutes): {round(time_, 4)}")
		test_times[test] = {"time": round(time_, 4), "moves": moves}

	print(test_times)

if __name__ == "__main__":
	# print(e.get_moves())
	# files = ["./benchmarks/"+f for f in listdir("./benchmarks/") if isfile(join("./benchmarks/", f)) and "sokoban" in f]
	# files.sort()
	test_model(["./benchmarks/sokoban-02.txt"])
	# play()