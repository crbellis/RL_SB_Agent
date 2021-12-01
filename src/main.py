from collections import deque
import time

from tensorflow.python.ops.gen_batch_ops import batch
from environment import Environment
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from DQN import agent
import tensorflow as tf
import random
# from path_finder import heuristic


from train import train as train2
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
	print(e.get_moves())
	state = e.to_int()
	state /= 5
	while move != "q" and not e.terminal():
		steps = 0
		print(e.to_int())
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
		print(action_set[np.argmax(pred)])
		move = input("Enter move (q to quit): ").lower()
		if move in moveMap.keys():
			prevCoords = e.player
			moves.append(move)


			prev_state = deepcopy(state)
			reward, done = e.move(moveMap[move])
			print(reward)
			state = e.to_int()
			state /= 5
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

def inspect(model, target_model):
	file = "./input/sokoban01.txt"
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
	state = e.to_int()
	state /= 5
	while move != "q" and not e.terminal():
		steps = 0
		print(e.to_int())
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
			state = e.to_int()
			state /= 5
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
	lr = 0.7
	gamma = 0.2
	
	MIN_REPLAY_SIZE = 1000
	if len(replay) < MIN_REPLAY_SIZE:
		return
	
	batch_size = 1000
	# random batch sampling - helps speed up training
	mini_batch = random.sample(replay, batch_size)
	# get current states
	current_states = np.array([transition[0] for transition in mini_batch])
	# get current states and Q-values
	current_q = model.predict(current_states.reshape(batch_size, 1, size))
	# get next states and Q-values
	new_states = np.array([transition[3] for transition in mini_batch])
	future_qs = target_model.predict(new_states.reshape(batch_size, 1, size))
	X=[]
	Y=[]
	
	# loop through sample and calculate Q-value using Bellman Equation
	for i, (state, action, reward, new_state, done) in enumerate(mini_batch):
		if not done:
			expected_state_action_value = reward + (gamma * np.max(future_qs[i][0]))

		else:
			expected_state_action_value = reward
		
		current_qs = current_q[i][0]
		# print("EXPECTED: ", expected_state_action_value, end="\t")
		# current_qs[action_idx[action]] = ((1-lr) * current_qs[action_idx[action]]) + (lr * expected_state_action_value)
		current_qs[action_idx[action]] = current_qs[action_idx[action]] + (lr * (expected_state_action_value - current_qs[action_idx[action]]))
		# print("CURRENT Q: ", current_qs[action_idx[action]])

		X.append(state.reshape(1, size))
		Y.append(current_qs.reshape(1, 4))
	
	Y = np.array(Y)
	X = np.array(X)
	# fit model to Q-values
	history = model.fit(X, Y, batch_size=batch_size, verbose=0, shuffle=True)
	return history.history


# testing model
def model_test():
	"""
	Model is trained using 2 neural networks, these networks use the same 
	architecture but have different weights. We can specify when the weights 
	from the model network are transferred to the target model

	Output of the models is a 4 element array - a Q-value for each of the
	possible actions. The best action is the one with the highest Q-value
	"""
	try:
		start = time.time()
		action_set = {0: "u", 1: "r", 2: "d", 3:'l'}
		action_idx = {"u":0, "r": 1, "d": 2, "l": 3}
		epsilon = 1 # Epsilon-greedy algorithm, initialized to 1 so every step is random to begin with 
		max_epsilon = 1
		min_epsilon = 0.2 # minimum always explore with 1% probability
		decay = 0.02 # rate of decay for epislon
		moveLimit = 300

		# instantiated environment
		file = "./input/sokoban01.txt"
		e = Environment()
		e.read_file(file)
		# print(heuristic("m", e.boxes, e.storage))
		# get input data for DQN model
		size = e.height * e.width
		# create two models, agent and target model
		model = agent((None, size), 4)
		target_model = agent((None, size), 4)
		target_model.set_weights(model.get_weights())

		# replay buffer to store previous experiences
		replay_buffer = deque(maxlen=50000)

		# training variables
		rewards = []
		steps_to_update_t_m = 0
		epochs = 0
		acc = []
		avg_acc = []
		loss = []
		avg_loss = []
		isTerminal = False
		while(not isTerminal and epochs < 300):
			epochs += 1
			moves = []
			total_R = 0
			done = False
			step = 0
			# reset from file
			e.read_file(file)
			state = e.to_int()
			state /= 5
			while(not done and len(moves) < moveLimit):
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
					predicted = model.predict(state.reshape(1, 1, size)).flatten()
					for i in range(4):
						if i not in valid_idx:
							predicted[i] = float("-inf")

					action = np.argmax(predicted)
				prev_state = deepcopy(state)
				reward, done = e.move(action_set[action])	

				# new state
				state = e.to_int()
				state /= 5

				moves.append(action_set[action])

				# experience replay - used to store previous game states
				replay_buffer.append([prev_state, action_set[action], reward, state, done])

				# update the model every 4 steps
				if step % 4 == 0 or done:
					history = train(replay_buffer, model, target_model, done, size)
					if history:
						acc.append(sum(history["accuracy"])/len(history["accuracy"]))
						loss.append(sum(history["loss"])/len(history["loss"]))

				total_R += reward

				step += 1

				if done:
					# update target model weights every 100 steps
					if steps_to_update_t_m >= 50:
						print("Copying main model weights to target model")
						target_model.set_weights(model.get_weights())
						steps_to_update_t_m = 0
					break

			boxCount = 0
			for box in e.boxes:
				if box in e.storage:
					boxCount += 1

			print(f"{epochs}: reward: {total_R:.0f}. (epsilon: {epsilon:.2f}).", 
				f" No. of boxes in storage: {boxCount}/{len(e.boxes)}.",
				f" No. of moves: {len(moves)}")

			epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * epochs) # change to epoch 

			if len(rewards) > 0 and (total_R > max(rewards)):
				print(f"REWARD: {total_R:.0f}\nBEST RESULT SO FAR: ")
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
			
		
		# end = time.time()

		# if len(rewards) < epochs:
		# 	for i in range(len(rewards), epochs):
		# 		rewards.append(0)

		# print("MINUTES TO RUN: ", (end - start)/60)
		# plt.plot(range(1, epochs+1), rewards)
		# plt.xlabel("Epoch")
		# plt.ylabel("Reward")
		# plt.show()
		
		# if len(avg_loss) > 0:
		# 	plt.clf()
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
		

	except KeyboardInterrupt:
		print("interupted")

	end = time.time()

	if len(rewards) < epochs:
		for i in range(len(rewards), epochs):
			rewards.append(0)

	print("MINUTES TO RUN: ", (end - start)/60)
	plt.plot(range(1, epochs+1), rewards)
	plt.xlabel("Epoch")
	plt.ylabel("Reward")
	plt.show()
	
	if len(avg_loss) > 0:
		plt.clf()
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
	inspect(model, target_model)	


if __name__ == "__main__":
	# main()
	model_test()
