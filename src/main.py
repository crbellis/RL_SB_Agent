from environment import Environment
import numpy as np
from model import model
import tensorflow as tf
import matplotlib.pyplot as plt
import path_finder
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
	# e.pretty_print()
	print(e.board)
	print(e.board.size * e.board.itemsize)

	for box in e.boxes:
		states = e.get_states(box)
		for state in states:
			print(state)
		print("GET STATES IN BYTES: ", state.size * state.itemsize)
	
	# print(e.board)
	# while move != "q" and not e.terminal():
	# 	e.pretty_print()
	# 	print(e.parseActions())
	# 	move = input("Enter move (q to quit): ").lower()
	# 	if move in moveMap.keys():
	# 		moves.append(move)
	# 		reward = e.move(moveMap[move])
	# 		print("REWARD: ", reward)
	# if e.terminal():
	# 	e.pretty_print()
	# 	print("You won! Here were your moves: ", moves)
	# else:
	# 	print("you quit")

if __name__ == "__main__":
	main()
