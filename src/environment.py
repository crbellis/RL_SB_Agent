from os import terminal_size


EMPTY = 0
PLAYER = "@"
STORAGE = "."
BOXES = "$"
WALLS = "#"
IN_STORAGE = "*"



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
		self.board = []

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
					self.player = data
				lineNum += 1

			# initializing empty board
			self.board = []
			for _ in range(self.height):
				self.board.append([0] * self.width)
			# plot elements
			self.plot(self.walls, WALLS)
			self.plot(self.boxes, BOXES)
			self.plot(self.storage, STORAGE)
			self.board[self.player[1]-1][self.player[0]-1] = PLAYER

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

	def parseActions(self) -> list:
		"""
		Finds all valid actions given current player state
		
		Returns
		-------
		validActions: list - all possible valid actions at current state
		"""
		validActions = []
		for action in self.movements.keys():
			coords = []
			if action in ("u", "d"):
				coords = [self.player[0], self.player[1]+self.movements[action]]
			else:
				coords =[self.player[0]+self.movements[action], self.player[1]] 
			if self.isValid(coords, action):
				validActions.append(action)
			
		return validActions

	def boxDetection(self, coords: list, row_or_col: int, move: str) -> bool:
		"""
		returns bool -  if valid collision
		"""
		# check for box collision
		if coords in self.boxes:
			# get the index where collision
			bIdx = self.boxes.index(coords)
			boxCoords = self.boxes[bIdx].copy()
			boxCoords[row_or_col] += self.movements[move]

			# check if moving the box is valid e.g. not pushed beyond wall
			if self.isValid(boxCoords, move):
				value = BOXES
				# check if box is in storage location
				if boxCoords in self.storage:
					value = IN_STORAGE
				self.board[boxCoords[1]-1][boxCoords[0]-1] = value
				self.boxes[bIdx] = boxCoords
			else:
				return False
		return True

	def terminal(self) -> bool:
		return all(coords in self.boxes for coords in self.storage)

	def move(self, move: str=None) -> bool or None:
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
		# check if move is in allowed moves
		assert move in ("u", "l", "r", "d")

		# get new coords
		newCoords = self.player.copy()
		# 0 for row and 1 for col
		row_or_col = 0
		if move in ("u", "d"):
			row_or_col = 1
		newCoords[row_or_col] += self.movements[move]

		# check if move isValid
		if self.isValid(newCoords, move):

			# check if player hit box i
			if self.boxDetection(newCoords, row_or_col, move):
				oldState = EMPTY
				# update current player location to empty space
				if self.player in self.storage:
					oldState = STORAGE

				self.board[self.player[1]-1][self.player[0]-1] = oldState
				# update player coords
				self.player = newCoords
				# update board with new player location
				self.board[self.player[1]-1][self.player[0]-1] = PLAYER
		if self.terminal():
			return True
		return

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
		return [[x, y] for x, y in zip(values[1::2], values[::2])]
		
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
			self.board[coord[1]-1][coord[0]-1] = symbol
	
	def pretty_print(self) -> None:
		"""
		Helper function to visualize board at current state
		"""
		# print out board
		for row in self.board:
			for column in row:
				print(column, end="")
			print()
		print()
