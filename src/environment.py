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
		self.movements = {"u": -1, "d": 1, "r": 1, "l": -1}
		self.height = height
		self.width = width
		self.walls = walls
		self.boxes = boxes
		self.storage = storage
		self.nWalls = nWalls
		self.nStorage = nStorage
		self.nBoxes = nBoxes
		self.player = player
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
			self.plot(self.walls, 4)
			self.plot(self.boxes, 3)
			self.plot(self.storage, 2)
			self.board[self.player[1]-1][self.player[0]-1] = 1

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
		if move in ("d", "u"):
			valid = valid and moveLimits[move] >= coords[1]-1
		else:
			valid = valid and moveLimits[move] >= coords[0]-1
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
			if action in ("u", "d"):
				if self.isValid([self.player[0], 
						self.player[1]+self.movements[action]], action):
					validActions.append(action)
			else:
				if self.isValid([self.player[0]+self.movements[action], 
						self.player[1]], action):
					validActions.append(action)
		return validActions

	def move(self, move: str) -> None:
		"""
		Moves the players coordinates in a given direction

		Parameters
		----------
		move: str - "u" | "d" | "l" | "r" e.g. direction in which the player is 
		to move

		Updates the player data member in [x, y] format
		"""
		# check if move is in allowed moves
		assert move in ("u", "l", "r", "d")
		# dictionary for values corresponding to each movement

		# get new coords
		if move in ("u", "d"):
			newCoords = [self.player[0], self.player[1]+self.movements[move]] 
		else:
			newCoords = [self.player[0]+self.movements[move], self.player[1]] 

		if self.isValid(newCoords, move):
			# update current player location to empty space
			self.board[self.player[1]-1][self.player[0]-1] = 0
			# update player coords
			self.player = newCoords
			# update board with new player location
			self.board[self.player[1]-1][self.player[0]-1] = 1
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
