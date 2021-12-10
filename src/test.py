from environment import Environment
from DQN import agent
from collections import deque
import numpy as np
from copy import deepcopy
from main import train
import matplotlib.pyplot as plt

plt.style.use("ggplot")

def play():
	"""
	format of input file:
	line0 = sizeH sizeV
	line1 = number of boxes and coordinates
	line2 = number of boxes and coordinates
	line3 = number of storage locations and coordinates
	line4 = player's starting coordinates =
	"""
	file = "./benchmarks/sokoban01.txt"

	# deadSpaces = [[1,1], [1,2], [1, 3], [1, 4]]
	e = Environment()
	e.read_file(file)
	print("ENV: ")
	e.pretty_print()
	# e.deadSpaces = deadSpaces
	moveMap = {"w": "u", "d": "r", "a": "l", "s": "d"}
	move = ""
	moves = []

	size = e.height * e.width
	# create two models, agent and target model
	model = agent((None, size), 4)
	target_model = agent((None, size), 4)
	target_model.set_weights(model.get_weights())

	
	order = e.getOrder2()
	print("ORDER: ", order)
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
def inspect(model, target_model, file, e = None):
	if e == None:
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
	order = e.getOrder()
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

def plot(epochs, rewards, avg_loss, avg_acc):
	"""
	Plots rewards, accuracy, and avg_loss
	"""
	if len(rewards) < epochs:
		for i in range(len(rewards), epochs):
			rewards.append(0)

	plt.plot(range(1, epochs+1), rewards)
	plt.xlabel("Epoch")
	plt.ylabel("Reward")
	plt.show()
	plt.clf()
	if len(avg_loss) > 0:
		plt.plot(range(1, len(avg_loss)+1), avg_loss)
		plt.xlabel("Epoch")
		plt.ylabel("Average Loss")
		plt.title("Avg. Loss per Epoch")
		plt.show()
		plt.clf()

		plt.plot(range(1, len(avg_acc)+1), avg_acc)
		plt.xlabel("Epoch")
		plt.ylabel("Average Accuracy")
		plt.show()
		plt.clf()

