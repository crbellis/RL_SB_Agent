from environment import Environment
import numpy as np
from model import model
import tensorflow as tf
import matplotlib.pyplot as plt
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
	# print(e.board)
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

if __name__ == "__main__":
	main()
