from re import I
from matplotlib.pyplot import box
import numpy as np
from path_finder import path

EMPTY = 0
PLAYER = 1 #"@"
STORAGE = 2 #"."
BOXES = 3 #"$"
WALLS = 4 #"#"
IN_STORAGE = 5 #"*"
PLAYER_IN_STORAGE = 6 #"+"

class Environment:
	# Environment constructor
	def __init__(
		self, height: int = 0, width: int = 0, walls: list = [], 
		boxes: list = [],storage: list = [], nWalls: int = 0, nStorage: int = 0, 
		nBoxes: int = 0, player: list = [], board: list = []
	):
		"""
		Constructor for Environment class
		coordinates are stored as [x, y] or [col, row]
		input is: N row1 col1... rowN colN
		0, 0 is top left of matrix
		"""
		# hash for all movements and corresponding change in position
		self.movements = {"u": -1, "r": 1,"d": 1, "l": -1}
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
		if len(board) > 0:
			self.board = np.array(board, dtype="bytes")
			self.player = np.argwhere(self.board == b'1').tolist()
			if len(self.player) == 0:
				self.player = np.argwhere(self.board == b'6').tolist()[0]
			else:
				self.player = self.player[0]
			self.player = [self.player[1], self.player[0]]
			self.storage = [[col, row] for row, col in np.argwhere(self.board == b'2').tolist()]
			self.boxes = [[col, row] for row, col in np.argwhere(self.board == b'3').tolist()]
			self.walls = [[col, row] for row, col in np.argwhere(self.board == b'4').tolist()]
		else:
			self.board = np.array([], dtype="bytes")
		self.totalReward = 0
		# used for undo
		self.movedBox = False


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
					self.player = [data[1]-1, data[0]-1]

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
				newCoords = coords.copy()
				newCoords[1] += self.movements[action]
			else:
				coords = [object[0]+self.movements[action], object[1]] 
				newCoords = coords.copy()
				newCoords[0] += self.movements[action]
			
			if self.isValid(coords, action):
				if coords in self.boxes:
					if newCoords in self.walls:
						continue

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
			self.board[newBox[1]][newBox[0]] = value
			# update boxes at boxIdx with new box coords
			self.boxes[boxIdx] = newBox
			return True
		return False

	def terminal(self) -> bool:
		return all(coords in self.boxes for coords in self.storage)
	
	def distance(self, coord1, coord2):
		return ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**0.5
		
	def find_nearest_storage(self, box) -> list:
		"""
		Finds the nearest storage unit

		Parameters
		----------
		box - list of col, row

		Returns
		-------
		Storage unit nearest the box 
		"""
		coords = []
		minDistance = float("inf")
		for storage in self.storage:
			dist = self.distance(box, storage)
			if dist < minDistance:
				if storage not in self.boxes:
					minDistance = dist
				coords = storage
		return coords

	def find_nearest_box(self) -> list:
		"""
		Finds nearest box to player

		Returns
		------
		list - coordinates for closest box
		"""
		coords = []
		minDist = float("inf")
		for box in self.boxes:
			dist = self.distance(self.player, box)
			if dist < minDist and box not in self.storage:
				minDist = dist
				coords = box
		return coords
	
	def isDeadLocked(self, boxCoords) -> bool:
		"""
		Checks if board is in deadlock state
		"""
		# check if walls surrounding left or right and top or bottom
		if (([boxCoords[0]+1, boxCoords[1]] in self.walls 
		or [boxCoords[0]-1, boxCoords[1]] in self.walls) 
		and ([boxCoords[0], boxCoords[1]+1] in self.walls 
		or [boxCoords[0], boxCoords[1]-1] in self.walls)):
			return True 

		elif (([boxCoords[0]+1, boxCoords[1]] in self.boxes
		or [boxCoords[0]-1, boxCoords[1]] in self.boxes) 
		and ([boxCoords[0], boxCoords[1]+1] in self.boxes 
		or [boxCoords[0], boxCoords[1]-1] in self.boxes)):	
			return True

		return False
		
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

		deadLocked = False
		self.movedBox = False
		reward = -1
		# check if move is in allowed moves
		assert move in ("u", "l", "r", "d")

		# get new coords
		newCoords = []
		if coords == None:
			newCoords = self.player.copy()
		else:
			newCoords = coords
		# 0 for col and 1 for row
		row_or_col = 0
		if move in ("u", "d"):
			row_or_col = 1
		
		nearestBox = self.find_nearest_box()
		current_distance = self.distance(self.player, nearestBox)
		newCoords[row_or_col] += self.movements[move]
		new_distance = self.distance(newCoords, nearestBox)
		
		if new_distance < current_distance:
			reward = 0

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
					closestStr = self.find_nearest_storage(boxCoords)
					# reward = 100 
					dist_from_new_box_coords = self.distance(boxCoords, closestStr)
					dist_from_old_box_coords = self.distance(newCoords, closestStr)
					if (dist_from_new_box_coords < dist_from_old_box_coords + 1):
						reward = 10 / max(self.distance(boxCoords, closestStr), 1)
					
					self.movedBox = True
					# update current player location to empty space
					if boxCoords in self.storage:
						boxes = np.array(self.boxes)
						storage = np.array(self.storage)
						nrows, ncols = boxes.shape
						dtype={'names':['f{}'.format(i) for i in range(ncols)],
	   						'formats':ncols * [boxes.dtype]}
						reward = 500 * (len(np.intersect1d(boxes.view(dtype), storage.view(dtype)))/len(self.boxes))
					else:
						# check for naive deadlocks
						if self.isDeadLocked(boxCoords):
								reward = -300
								deadLocked = True
				else:
					return reward, False
					
			if self.player in self.storage:
				oldState = STORAGE

			self.board[self.player[1]][self.player[0]] = oldState

			# update player coords
			self.player = newCoords
			# update board with new player location
			newState = PLAYER
			if self.player in self.storage:
				newState = PLAYER_IN_STORAGE

			self.board[self.player[1]][self.player[0]] = newState

		self.totalReward += reward
		if self.terminal():
			return reward + 1000, True
		return reward, False or deadLocked

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
		# col or x is second value in list, hence starting at 1
		# 0 indexing so subtracting 1 from the value
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
		for coord in coords:
			if symbol == 2:
				if coord in self.boxes:
					self.board[coord[1]][coord[0]] = IN_STORAGE
					continue

			self.board[coord[1]][coord[0]] = symbol
		return	
	
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
					if [_column, _row] in self.storage:
						print("+", end="")
					else:
						print("@", end="")
				elif ([_column, _row] in self.boxes 
					and [_column, _row] in self.storage):
					print("*", end="")
				elif [_column, _row] in self.boxes:
					print("$", end="")
				elif [_column, _row] in self.storage:
					print(".", end="")
				else:
					print(" ", end="")
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

	def to_float(self):
		"""
		Converts state to float based numpy array
		"""
		return np.array(self.board, dtype="float32")
	
	def get_moves(self):
		"""
		Returns all possible box states from current state based on reachable
		paths

		Coords are in [x, y] format 0 indexed where 0,0 is top left of matrix
		returns: [[[x, y], a].... [[x, y], a]]
		"""
		# convert state to boolean format for path finder
		bool_state = np.where(self.board == b'4', False, True)
		inverseLoc = {"l": "r", "r": "l", "u": "d", "d": "u"}
		actions = []
		for box in self.boxes:
			try:
				# get ways box can move
				path_to_box = path(
					(self.player[1], self.player[0]), 
					(box[1], box[0]), 
					100, 
					bool_state
				)
				if path_to_box != None:
					validActions = self.parseActions(box)
					# loop through valid actions
					for action in validActions:
						# set the coordinates to current box
						coord = box
						pushedBox = coord

						# get the inverse location i.e. if moving up need to be able to access down location
						inverse = inverseLoc[action]
						if action in ("d", "u"):
							pushedBox = [coord[0], coord[1] + self.movements[action]]
							coord = [coord[0], coord[1] + self.movements[inverse]]
						else:
							pushedBox = [coord[0]+self.movements[action], coord[1]]
							coord = [coord[0]+self.movements[inverse], coord[1]]
						if coord not in self.walls and coord not in self.boxes and pushedBox not in self.boxes:
							actions.extend([box, action])
			except:
				pass
		output = []
		for i in range(int(len(actions)/2)):
			output.append((actions[2*i],actions[2*i+1])) 
		return output

	def undo(self, movement: list) -> None:
		"""
		This function undos a movement which is passed as an argument.
		The expected input is the previous coordinate along with the action 
		taken. This happens in-place on the current instantiated board.

		i.e. if the player was in [2, 3] and moved up to [2, 2] then this 
		function will move the player back down to [2, 3]. If a box was pushed 
		when moving, the box will be reset as well

		Parameters
		----------
		movement - list of coordinates [x, y] (0 indexed) and move: 
		str "u" | "d" | "l" | "r"
		"""
		action = movement[1]
		prevPos = movement[0]
		currentPos = prevPos
		actionValue = self.movements[action]
		tempBox = []
		print(movement)
		if action in ("u", "d"):
			currentPos = [currentPos[0], currentPos[1]+actionValue]
			tempBox = [currentPos[0], currentPos[1]+actionValue]
		else:
			currentPos = [currentPos[0]+actionValue, currentPos[1]]
			tempBox = [currentPos[0]+actionValue, currentPos[1]]
		if self.movedBox:
			index = self.boxes.index(tempBox)
			self.boxes[index] = currentPos 
		self.player = prevPos
		
	def move_block(self, coordinate, direction):
		old_player_x, old_player_y = self.player[0], self.player[1]
		old_player_val = self.board[old_player_y, old_player_x]
		coord = coordinate.copy()
		index = -1
		for i in range(len(self.boxes)):
			if self.boxes[i] == coord:
				index = i
		if index == -1:
			raise("Invalid coord error")
			return
		if direction == 'u':
			self.player = coord
			self.boxes[index][1]-=1
		elif direction == 'd':
			self.player = coord
			self.boxes[index][1] +=1
		elif direction == 'l':
			self.player = coord
			self.boxes[index][0] -=1
		elif direction == 'r':
			self.player = coord
			self.boxes[index][0] +=1
		else:
			print("Invald direction error")

		# updates where the block used to be and where the player currently is
		if self.player in self.storage:
			self.board[self.player[1],self.player[0]] = b'6'
		else:
			self.board[self.player[1],self.player[0]] = b'1'
		
		# updates where the player used to be in self.board
		if old_player_val == b'1':
			self.board[old_player_y, old_player_x] = b'0'
		else:
			self.board[old_player_y, old_player_x] = b'2'
		
		# update where the block is now
		if self.boxes[index] in self.storage:
			self.board[self.boxes[index][1], self.boxes[index][0]] = b'5'
		else:
			self.board[self.boxes[index][1], self.boxes[index][0]] = b'3'
		return self.boxes[index]
	
	def undo_move_block(self, coord, direction):
		player_coord = coord.copy()
		original_coord = coord.copy()
		undone_coord = coord.copy()
		if direction == 'u':
			player_coord[1]+=1
			undone_coord[1]-=1
			self.player= player_coord
		elif direction == 'd':
			player_coord[1] -=1
			undone_coord[1] +=1
			self.player= player_coord
		elif direction == 'l':
			player_coord[0] +=1
			undone_coord[0] -=1
			self.player= player_coord
		elif direction == 'r':
			player_coord[0] -=1
			undone_coord[0] +=1
			self.player= player_coord
		else:
			print("Invald direction error")
		index = -1
		
		for i in range(len(self.boxes)):
			if self.boxes[i] == undone_coord:
				index = i
		if index == -1:
			print("Invalid coord error on undo")
			return
		self.boxes[index] = original_coord
		
		# update the now blank square where the block was, pre undo
		if undone_coord in self.storage:
			self.board[undone_coord[1], undone_coord[0]] = b'2'
		else:
			self.board[undone_coord[1], undone_coord[0]] = b'0'
		
		# updates block position after undo
		if original_coord in self.storage:
			self.board[original_coord[1], original_coord[0]] = b'5'
		else:
			self.board[original_coord[1], original_coord[0]] = b'3'
			
		# update player position after undo
		if self.player in self.storage:
			self.board[self.player[1], self.player[0]] = b'6'
		else:
			self.board[self.player[1], self.player[0]] = b'1'
		return
