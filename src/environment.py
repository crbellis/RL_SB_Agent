import numpy as np
from path_finder import path

EMPTY = 0
PLAYER = 1 #"@"
STORAGE = 2 #"."
BOXES = 3 #"$"
WALLS = 4 #"#"
IN_STORAGE = 5 #"*"

class Environment:
	# Environment constructor
	def __init__(
		self, height: int = 0, width: int = 0, walls: list = [], 
		boxes: list = [],storage: list = [], nWalls: int = 0, nStorage: int = 0, 
		nBoxes: int = 0, player: list = []
	):
		"""
		Constructor for Environment class
		"""
		# hash for all movements and corresponding change in position
		self.movements = {"u": -1, "d": 1, "r": 1, "l": -1}
		# data from input
		self.height = height
		self.width = width
		self.walls = walls
		self.boxes = boxes
		self.storage = storage
		self.nWalls = nWalls
		self.nStorage = nStorage
		self.nBoxes = nBoxes
		self.player = player
		# entire board state representation 2x2 list of size height x width
		self.board = np.array([], dtype="bytes")
		self.totalReward = 0


	# initialize environment from file
	def read_file(self, path: str = "") -> None:
		"""
		Parses data from input text file. Initializes object from file data

		Parameters:
		----------
		path: str - path to input file
		"""
		try:
			file = open(path)
			lineNum = 0
			for line in file:
				data = line.split(" ")
				data = [int(x) for x in data] # cast values to int
				# board size
				if lineNum == 0:
					self.height = data[0]
					self.width = data[1]
				# walls
				elif lineNum == 1:
					self.nWalls = data[0]
					# set walls to [x, y] pairs
					self.walls = self.zip_coords(data[1:]) 
				# boxes
				elif lineNum == 2:
					self.nBoxes = data[0]
					self.boxes = self.zip_coords(data[1:])
				# storage locations
				elif lineNum == 3:
					self.nStorage = data[0]
					self.storage = self.zip_coords(data[1:])
				# start coords
				else:
					self.player = [data[0]-1, data[1]-1]

				lineNum += 1

			# initializing empty board
			self.board = []
			for _ in range(self.height):
				self.board.append([0] * self.width)
			# plot elements
			self.plot(self.walls, WALLS)
			self.plot(self.boxes, BOXES)
			self.plot(self.storage, STORAGE)
			self.board[self.player[1]][self.player[0]] = PLAYER
			self.board = np.array(self.board, dtype="bytes")

		except Exception as e:
			print(e)


	def isValid(self, coords, move) -> bool:
		"""
		Checks if move is into a wall or if move is out of bounds of environment

		Parameters:
		-----------
		coords: list in [x, y] format starting at index 1
		move: "u" | "d" | "l" | "r"

		Returns:
		-------
		isValid: bool - if move coordinates + move are valid
		"""
		moveLimits = {
			"d": len(self.board)-1, 
			"u": len(self.board)-1, 
			"l": len(self.board[self.player[1]-1]) - 1,
			"r": len(self.board[self.player[1]-1]) - 1
		}
		valid = coords not in self.walls
		coord = 0

		if move in ("d", "u"):
			coord = coords[1]-1
		else:
			coord = coords[0]-1

		valid = valid and moveLimits[move] >= coord
		# if not valid:
		# 	print("out of bounds")
		return valid

	def parseActions(self, object: list=None) -> list:
		"""
		Finds all valid actions given current player state
		
		Returns
		-------
		validActions: list - all possible valid actions at current state
		"""
		if object == None:
			object = self.player
		validActions = []
		for action in self.movements.keys():
			coords = []
			if action in ("u", "d"):
				coords = [object[0], object[1]+self.movements[action]]
			else:
				coords =[object[0]+self.movements[action], object[1]] 
			if self.isValid(coords, action):
				validActions.append(action)
		return validActions

	def boxDetection(self, newBox: list, boxIdx: int, move: str) -> bool:
		"""
		returns bool -  if valid collision or if not a box
		"""
		# check if moving the box is valid e.g. not pushed beyond wall
		if self.isValid(newBox, move) and newBox not in [box for i,box in enumerate(self.boxes) if i != boxIdx]:
			value = BOXES
			# check if box is in storage location
			if newBox in self.storage:
				value = IN_STORAGE

			# update board with new value	
			self.board[newBox[1]-1][newBox[0]-1] = value
			# update boxes at boxIdx with new box coords
			self.boxes[boxIdx] = newBox
			return True
		return False

	def terminal(self) -> bool:
		return all(coords in self.boxes for coords in self.storage)

	# TODO: edit block movement rewards to use heuristic from path_finder
	def move(self, move: str=None, coords: list = None) -> tuple[int, bool]:
		"""
		Moves the players coordinates in a given direction

		Parameters
		----------
		move: str - "u" | "d" | "l" | "r" e.g. direction in which the player is 
		to move
		objectCoords: list - list of [x, y] coordinates for object. Intended 
		objects are those that can move e.g. player and box objects

		Updates the player data member in [x, y] format
		"""
		reward = -1
		# check if move is in allowed moves
		assert move in ("u", "l", "r", "d")

		# get new coords
		newCoords = []
		if coords == None:
			newCoords = self.player.copy()
		else:
			newCoords = coords
		# 0 for row and 1 for col
		row_or_col = 0
		if move in ("u", "d"):
			row_or_col = 1

		newCoords[row_or_col] += self.movements[move]

		# check if move isValid
		if self.isValid(newCoords, move):
			oldState = EMPTY

			# check if player hit box and if it's valid to move the box
			# check for box collision
			if newCoords in self.boxes:

				# update the box with new coords
				# get the index where collision
				bIdx = self.boxes.index(newCoords)
				boxCoords = self.boxes[bIdx].copy()
				boxCoords[row_or_col] += self.movements[move]

				# check if box is valid detection
				if self.boxDetection(boxCoords, bIdx, move):
					# reward = 100 
					reward = 0
					# update current player location to empty space
					if boxCoords in self.storage:
						reward = 1000
				else:
					return reward, False

			if self.player in self.storage:
				oldState = STORAGE

			self.board[self.player[1]-1][self.player[0]-1] = oldState

			# update player coords
			self.player = newCoords
			# update board with new player location
			self.board[self.player[1]-1][self.player[0]-1] = PLAYER

		self.totalReward += reward
		if self.terminal():
			return reward, True
		return reward, False

	def zip_coords(self, values: list) -> list:
		"""
		Combines list in to [x, y] coordinates

		Parameters
		----------
		values: list - a list of values, parsed from input text file

		Returns
		-------
		coordinates: list - list of coordinates e.g. [[x1, y1], [x2, y2]...]
		"""
		return [[x-1, y-1] for x, y in zip(values[1::2], values[::2])]
		
	def plot(self, coords: list, symbol: str or int) -> None:
		"""
		Helper function to set coords to a given symbol

		Parameters
		----------
		coords: list - list of coordinates to update representation
		symbol: str | int - symbol to represent the given coords
		"""
		# coord[1] -> row (y)
		# coord[0] -> column (x) hence backwards indexing
		# -1 since 0 indexed and coords are cartesian
		for coord in coords:
			self.board[coord[1]][coord[0]] = symbol
	
	def pretty_print(self) -> None:
		"""
		Helper function to visualize board at current state
		"""
		# print out board
		_row = _column = 0
		for row in self.board:
			for column in row:
				if [_column, _row] in self.walls:
					print("#", end="")
				elif [_column, _row] == self.player:
					print("@", end="")
				elif ([_column, _row] in self.boxes 
					and [_column, _row] in self.storage):
					print("*", end="")
				elif [_column, _row] in self.boxes:
					print("$", end="")
				elif [_column, _row] in self.storage:
					print(".", end="")
				else:
					print("0", end="")
				_column += 1
			print()
			_row += 1
			_column = 0
		print()
	
	def to_int(self):
		"""
		Converts state to int based numpy array
		"""
		return np.array(self.board, dtype="int32")
	
	# TODO: update return to data structure below
	# TODO: extend moves function to take these objects as argument and make a move
	# Create an undo function that can take a coordinate and action
	# pass same x, y and a allow it to undo

	# [[(x, y), a].... [(x, y), a]]
	def get_moves(self) -> list:
		"""
		Returns all possible box states from current state based on reachable
		paths
		"""
		# convert state to boolean format for path finder
		bool_state = np.where(self.board == b'4', False, True)
		actions = []
		for box in self.boxes:
			try:
				path_to_box = path(
					(self.player[1]-1, self.player[0]-1), 
					(box[1]-1, box[0]-1), 
					100, 
					bool_state
				)
				if path_to_box != None:
					validActions = self.parseActions(box)
					actions.append(validActions)
			except:
				pass
		return actions
		# print("PATH: ", path_to_box)
		# states = []
		# box_pos = []
		# actions = self.parseActions(box)
		# for action in actions:
		# 	tempBox = box.copy()
		# 	if action in ("u", "d"):
		# 		tempBox[1] += self.movements[action]
		# 	else:
		# 		tempBox[0] += self.movements[action]
		# 	box_pos.append(tempBox)

		# for pos in box_pos:
		# 	state = np.copy(self.board)
		# 	state[box[1]-1][box[0]-1] = EMPTY # clearing box from state
		# 	state[pos[1]-1][pos[0]-1] = BOXES
		# 	states.append(state)
		# return states

	def undo(self):
		pass