class Environment:
	# Environment constructor
	def __init__(
		self, height: int = 0, width: int = 0, walls: list = [], 
		boxes: list = [],storage: list = [], nWalls: int = 0, nStorage: int = 0, 
		nBoxes: int = 0, player: list = []
	):
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
	def read_file(self, path: str = ""):
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
			self.plot(self.board, self.walls, "#")
			self.plot(self.board, self.boxes, "$")
			self.plot(self.board, self.storage, ".")
			self.board[self.player[1]-1][self.player[0]-1] = "@"

		except Exception as e:
			print(e)

	def move(self, move: str):
		# check if move is in allowed moves
		assert move in ("u", "l", "r", "d")
		# dictionary for values corresponding to each movement
		movements = {"u": -1, "d": 1, "r": 1, "l": -1}

		# get new coords
		if move in ("u", "d"):
			newCoords = [self.player[1], self.player[0]+movements[move]] 
		else:
			newCoords = [self.player[0]+movements[move], self.player[1]] 

		if self.isValid(newCoords, move):
			# update current player location to empty space
			self.board[self.player[1]-1][self.player[0]-1] = "_"
			# update player coords
			self.player = newCoords
			# update board with new player location
			self.board[self.player[1]-1][self.player[0]-1] = "@"
		return



	def isValid(self, coords, move):
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
		if not valid:
			print("out of bounds")
		return valid

	def zip_coords(self, values):
		return [[x, y] for x, y in zip(values[1::2], values[::2])]
		
	def plot(self, board: list, coords: list, symbol: str):
		# coord[1] -> row (y)
		# coord[0] -> column (x) hence backwards indexing
		# -1 since 0 indexed and coords are cartesian
		for coord in coords:
			board[coord[1]-1][coord[0]-1] = symbol
	
	def pretty_print(self):

		# print out board
		for row in self.board:
			for column in row:
				if column != 0 and column != "":
					print(column, end="")
				else:
					print("_", end="")
			print()
		print()
