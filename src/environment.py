import numpy as np
import grids
from path_finder import path, manhattan_distance
import copy
import pickle	
import numpy, time
import random

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
		nBoxes: int = 0, player: list = [], board: list = [],
		twos = None, threes = None, solvable_subproblems = None, saved_deadlocks = None
	):
		"""
		Constructor for Environment class
		coordinates are stored as [x, y] or [col, row]
		input is: N row1 col1... rowN colN
		0, 0 is top left of matrix
		"""
		
		if twos == None:
			self.twos, self.threes  = pickle.load(open("deadlocks", "rb"))
			self.solvable_subproblems = set()
			self.saved_deadlocks = set()
		else:
			self.twos, self.threes = twos, threes
			self.solvable_subproblems, self.saved_deadlocks = solvable_subproblems, saved_deadlocks
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
		self.totalReward = 0
		self.old_moves = 0
		self.candidateOrder = []
		self.candidateIdx = 0
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
			self.storage += [[col, row] for row, col in np.argwhere(self.board == b'5').tolist()]
			self.storage += [[col, row] for row, col in np.argwhere(self.board == b'6').tolist()]
			self.boxes = [[col, row] for row, col in np.argwhere(self.board == b'3').tolist()]
			self.boxes += [[col, row] for row, col in np.argwhere(self.board == b'5').tolist()]
			self.walls = [[col, row] for row, col in np.argwhere(self.board == b'4').tolist()]
			self.old_moves = len(self.get_moves())
		else:
			self.board = np.array([], dtype="bytes")

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
					if "-" not in path:
						self.height = data[0]
						self.width = data[1]
					else: 
						self.height = data[1]
						self.width = data[0]
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
			self.old_moves = len(self.get_moves())
			self.candidateIdx =0
		except Exception as e:
			print(e)

	# checks if it is possible to move to a coordinate
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
	

	def parseActions(self, object: list=None, astar = False) -> list:
		"""
		Finds all valid actions given current player state
		Does not allow moves that create 2x2 and 3x3 deadlocks
		
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

				temp = self.board.copy()	
				if object in self.storage:
					temp[object[1]][object[0]] = 2
				else:
					temp[object[1]][object[0]] = 0
				if coords in self.storage:
					temp[coords[1]][coords[0]] = 6
				else:
					temp[coords[1]][coords[0]] = 1

				if coords in self.boxes:
					if newCoords in self.storage:
						temp[newCoords[1]][newCoords[0]] = 5
					else:
						temp[newCoords[1]][newCoords[0]] = 3
					deadlock = self.check_deadlocks(temp)
					if deadlock:
						continue
					else:
						afterBlockCoords = newCoords.copy()
						if action in ("u", "d"):
							afterBlockCoords[1] += self.movements[action]
						else:
							afterBlockCoords[0] += self.movements[action]
						touchingWall = afterBlockCoords in self.walls
						if touchingWall and not astar:
							if not newCoords in self.storage:
								can_solve, hashable_board = Environment(board = temp, twos = self.twos, threes = self.threes,\
												solvable_subproblems=self.solvable_subproblems, saved_deadlocks=self.saved_deadlocks).solve_subproblem\
									(newCoords, self.solvable_subproblems, self.saved_deadlocks)
								if can_solve:
									if not hashable_board == None:
										self.solvable_subproblems.add(hashable_board)
								else:
									if not hashable_board == None:
										self.saved_deadlocks.add(hashable_board)
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
	
	# Checks for all 2x2 and 3x3 deadlocks given an array
	def check_deadlocks(self, arr):
		twos, threes = self.twos, self.threes
		twos.add(bytes((0,4,4,3)))
		twos.add(bytes((4,3,0,4)))
		twos.add(bytes((3,4,4,0)))
		twos.add(bytes((4,0,3,4)))
		twos.add(bytes((0,4,4,5)))
		twos.add(bytes((4,5,0,4)))
		twos.add(bytes((5,4,4,0)))
		twos.add(bytes((4,0,5,4)))
		two_by_two_sub_arrays = []
		for i in range(len(arr) - 1):
			for j in range(len(arr[0]) - 1):
				two_by_two_sub_arrays.append(arr[i:i+2, j:j+2])
		three_by_three_sub_arrays = []
		for i in range(len(arr) - 2):
			for j in range(len(arr[0]) - 2):
				three_by_three_sub_arrays.append(arr[i:i+3, j:j+3])
		for t in two_by_two_sub_arrays:
			if grids.make_hashable(t) in twos:
				return True
		for thr in three_by_three_sub_arrays:
			if grids.make_hashable(thr) in threes:
				return True
		return False
	
	# Checks for all 2x2 and 3x3 deadlocks given a coordinate of a moved box
	def isDeadLocked(self, boxCoords) -> bool:
		"""
		Checks if board is in deadlock state
		"""
		x, y = boxCoords[0], boxCoords[1]

		# change this to check for larger deadlocks in the future
		left_pad = 2
		right_pad = 3
		down_pad = 3
		up_pad = 2

		if x-2 < 0:
			left_pad -= 1
		if x+3 >= len(self.board[1]):
			right_pad -= 1
		if y-2 < 0:
			up_pad -= 1
		if y+3 >= len(self.board):
			down_pad -= 1

		arr = self.board[y-up_pad: y+down_pad, x-left_pad: x+right_pad]
		result = self.check_deadlocks(arr)
		return result

	# Moves to a block if a path exists
	def go_to(self, coord):
		path_to_coord = path(
			(self.player[1], self.player[0]),
			(coord[1], coord[0]),
			100,
			np.where((self.board == b'4') | (self.board == b'3') | (self.board == b'5'), False, True)
		)
		path_to_coord = [[x, y] for y, x in path_to_coord]
		movements = []
		prevCoord = []
		for movement in path_to_coord:
			compare = prevCoord
			if len(prevCoord) == 0:
				compare = self.player

			if movement[0] > compare[0]:
				movements.append("r")
			elif movement[0] < compare[0]:
				movements.append("l")
			
			if movement[1] > compare[1]:
				movements.append("d")
			elif movement[1] < compare[1]:
				movements.append("u")
			prevCoord=movement

		oldState = EMPTY
		if self.player in self.storage:
			oldState = STORAGE
		
		self.board[self.player[1]][self.player[0]] = oldState
		self.player = path_to_coord[-1]
		self.board[self.player[1]][self.player[0]] = PLAYER
		return movements

	def move(self, move: str=None, coords: list = None):
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

					self.movedBox = True
					# update current player location to empty space
					boxes = np.array(self.boxes)
					storage = np.array(self.storage)
					nrows, ncols = boxes.shape
					dtype={'names':['f{}'.format(i) for i in range(ncols)],
						'formats':ncols * [boxes.dtype]}
					
					if self.candidateIdx == 0:
						flag = False
					elif newCoords == self.candidateOrder[self.candidateIdx-1]:
						flag = True
					else:
						flag = False
					if boxCoords == self.candidateOrder[self.candidateIdx] and not flag:
						reward += 500 * (len(np.intersect1d(boxes.view(dtype), storage.view(dtype)))/len(self.boxes))
						self.candidateIdx += 1 
					elif flag:
						reward += -500 * ((len(np.intersect1d(boxes.view(dtype), storage.view(dtype))) + 1)/len(self.boxes))
						self.candidateIdx -= 1
						self.candidateIdx = max(0, self.candidateIdx)

					num_moves = len(self.get_moves())
					reward += (num_moves - self.old_moves) * 0.3
					reward += 5

					self.old_moves = num_moves

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

# 			if self.movedBox and boxCoords not in self.storage:
# 				if self.isDeadLocked(boxCoords):
# 					reward = -1000
# 					deadLocked=True

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
	
	
	def get_moves(self, astar = False):
		"""
		Returns all possible box states from current state based on reachable
		paths

		Coords are in [x, y] format 0 indexed where 0,0 is top left of matrix
		returns: [[[x, y], a].... [[x, y], a]]
		"""
		# convert state to boolean format for path finder
		bool_state = np.where((self.board == b'4') | (self.board == b'3') | (self.board == b'5'), False, True)
		inverseLoc = {"l": "r", "r": "l", "u": "d", "d": "u"}
		actions = []
		for box in self.boxes:
			# get ways box can move
			path_to_box = path(
				(self.player[1], self.player[0]), 
				(box[1], box[0]), 
				100, 
				bool_state
			)
			if path_to_box != None:
				validActions = self.parseActions(box, astar)
				# loop through valid actions
				for action in validActions:
# 					print('before',box, action, validActions)
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
# 						print('box = ', box, ' action = ', action, 'pushedBox = ', pushedBox, 'coord = ', coord)
					path_to_push_coord = path(
						(self.player[1], self.player[0]), 
						(coord[1], coord[0]), 100, bool_state
					)
					if not path_to_push_coord == None and len(path_to_push_coord) > 0 and \
						 coord not in self.walls and coord not in self.boxes and \
							  pushedBox not in self.boxes and action in self.parseActions(coord, astar):
						actions.extend([box, action])
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
	
	# Takes a block and a coordinate from get_moves output and has player move that box
	# Used in A*
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
	
	# Used in A* to undo a move
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
	
	
	# Uses A* to solve a subproblem 
	def solve_subproblem(self, box_coord, solved_subproblems, saved_deadlocks, max_time_per_problem = 1):
		b_x, b_y = box_coord[0], box_coord[1]
		p_x, p_y = self.player[0], self.player[1]
		if box_coord in self.storage:
			return True, None
		for s in self.storage:
			new_board = copy.deepcopy(self.board)
			for x,y in self.storage:
				new_board[y,x] = 0
			for x,y in self.boxes:
				new_board[y,x] = 0
			s_x, s_y = s[0], s[1]
			new_board[b_y, b_x] = 3
			new_board_identifier = grids.make_hashable(new_board)
			if new_board_identifier in solved_subproblems:
				return True, None
			elif new_board_identifier in saved_deadlocks:
				return False, None
			else:
				new_board[s_y, s_x] = 2
				new_board[p_y, p_x] = 1
				can_solve = A_star(Environment(board=new_board), max_time = max_time_per_problem).solve()
				if can_solve:
					return True, new_board_identifier
		return False, new_board_identifier
	
	# Determines a candidate order in which storages are filled
	def getOrder2(self):
		t0 = time.time()
		i = 1
		t = 5
		while True:
			if time.time() - t0 > 20*i:
				t0 = time.time()
				t += 5
				i += 1
			order = self.try_random_order(t)
			if order != None:
				self.candidateOrder = [self.storage[i[1]] for i in order]
				return self.candidateOrder
	
	# Uses A* to identify if a random order is a plausible candidate to fill storage locations
	def try_random_order(self, t):
		temp = copy.deepcopy(self.board)
		box_order = [i for i in range(len(self.boxes))]
		storage_order = box_order.copy()
		random.shuffle(box_order)
		random.shuffle(storage_order)
		random_matching = [[i,j] for i,j in zip(box_order, storage_order)]
		self.temp_boards = []
		
		#initializing reduced board
		for i in range(len(temp)):
			for j in range(len(temp[0])):
				if temp[i,j] == b'2' or temp[i,j] == b'3' or temp[i,j] == b'5' or temp[i,j] == b'6':
					temp[i,j] = b'0'
		for box_index, storage_index in random_matching:
			b_x, b_y = self.boxes[box_index]
			s_x, s_y = self.storage[storage_index]
			temp[b_y,b_x] = b'3'
			temp[s_y,s_x] = b'2'
			if not b'1' in temp:
				p_y,p_x = numpy.argwhere(temp == b'0')[0]
				temp[p_y,p_x] = b'1'
			tempE = Environment(board=numpy.copy(temp), twos = self.twos, threes = self.threes,
					solvable_subproblems=self.solvable_subproblems, saved_deadlocks=self.saved_deadlocks)
			a = A_star(tempE, max_time = t)
			if a.solve():
				self.temp_boards.append(temp)
				temp[b_y,b_x] = b'0'
				temp[s_y,s_x] = b'4'
			else:
				self.solvable_subproblems = set.union(self.solvable_subproblems, a.test_board.solvable_subproblems)
				self.saved_deadlocks = set.union(self.saved_deadlocks, a.test_board.saved_deadlocks)
				return None
		self.matching = [[self.storage[i[1]], self.boxes[i[0]]] for i in random_matching]
		return random_matching
	
	def setOrder(self, order):
		self.candidateOrder = order
		self.candidateIdx = 0

# A* seach node
class Node:
	
	# state is the starting state for the problem, before any movements
	def __init__(self, parent, move, depth):
		self.parent = parent
		self.move = move
		self.f = -1 # as in f(n) = g(n) + h(n)
		self.depth = depth
		# root is Node(None, None, 0)
		
	def list_moves(self):
		# gives sequence of moves up through node n
		output = []
		node = self
		while node.depth > 0:
			output.append(node.move)
			node = node.parent
		output.reverse()
		return output
	
import path_finder

# Conducts the A* search
class A_star:
	def __init__(self, starting_board, max_time = 5):
		self.starting_board = starting_board
		self.test_board = Environment(board=copy.deepcopy(starting_board.board), twos = starting_board.twos,
			threes = starting_board.threes, solvable_subproblems=starting_board.solvable_subproblems,
			saved_deadlocks=starting_board.saved_deadlocks) #this is the object we manipulate to check for heuristic values
		self.frontier = [Node(None, None, 0)]
		self.moveable_squares_matrix = list(map(list, zip(*path_finder.get_moveable_squares_matrix(starting_board.to_int()))))
		self.seen = set()
		self.max_time = max_time
	
	
	# returns none sub problem cannot be solved
	def solve(self, is_three_by_three = False):
		t0 = time.time()
		self.frontier[0].f = self.get_h()
		if is_three_by_three:
			max_depth = 2 * self.frontier[0].f
		else:
			max_depth = 2**30
		while (len(self.frontier) != 0):
			if time.time() - t0 > self.max_time:
				return(False)

			n = self.frontier.pop()
			
			if n.depth >= max_depth:
				return False
			
			self.test_board = self.initialize_test_board(n)
			
			moves = copy.deepcopy(self.test_board.get_moves(astar=True))
			
			#iterate through the moves to create children for the best current node
			for movement in moves:
				
				
				move = movement[0].copy(),movement[1]
				new_node = Node(n, move, n.depth+1)

				# moves block and later restores new block position
				self.test_board.move_block(move[0],move[1])
				
				board_identifier = self.make_hashable()
				if not board_identifier in self.seen:
					self.seen.add(board_identifier)
					if not self.test_board.isDeadLocked(move[0]):
						# The get_h value is like h(n) ~ heuristic, depth is like g(h) ~ cost
						h = self.get_h()
						if h == 0:
							return True
						new_node.f = h + 0.5 * new_node.depth
						self.frontier.append(new_node)
					

				# restores new block position
				self.test_board.undo_move_block(move[0],move[1])
				
			# sort the frontier based on f
			self.frontier = [i for i,_ in sorted(zip(self.frontier, [-j.f for j in self.frontier]), key = lambda k: k[1])]
		return False

	
	# Makes numpy array a hashable object
	def make_hashable(self):
		output = self.test_board.board
		output = output.reshape((len(output) * len(output[0])))
		return tuple(output)
		
	# Returns heuristic value using minimal matching heuristic for manhattan distances
	def get_h(self):
		block_positions = self.test_board.boxes
		storage_positions = self.test_board.storage
		h = path_finder.heuristic('m', block_positions, storage_positions, self.moveable_squares_matrix)
		return h
	
	# Resets the board in A*
	def initialize_test_board(self, node):
		moves = node.list_moves()
		starting_board=self.starting_board
		testBoard = Environment(board=copy.deepcopy(starting_board.board), twos = starting_board.twos,
			threes = starting_board.threes, solvable_subproblems=starting_board.solvable_subproblems,
			saved_deadlocks=starting_board.saved_deadlocks)
		for move in moves:
			testBoard.move_block(move[0],move[1])
		return testBoard

# Returns approximation for size of the state space
def complexity_measure(e):
	from math import comb
	arr = copy.deepcopy(e.board)
	blocks = len(numpy.argwhere((arr == b'3') | (arr == b'5')))
	moveable_squares = len(numpy.argwhere((arr != b'4')))
	print('blocks = ', blocks, 'moveable_squares = ', moveable_squares)
	return comb(moveable_squares, blocks)


	
	