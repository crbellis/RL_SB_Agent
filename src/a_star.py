import psutil, copy, numpy, path_finder, grids, pickle, time, matplotlib.pyplot as plt

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
		
class A_star:
	
	def __init__(self, starting_board, max_time = 5):
		self.starting_board = starting_board
		self.test_board = copy.deepcopy(starting_board) #this is the object we manipulate to check for heuristic values
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
		
		#100MB free memory required
		v = 0
		while (len(self.frontier) != 0):
			if time.time() - t0 > self.max_time:
# 				print('timeout')
				return(False)
# 			if psutil.virtual_memory()[1] > 100 * 2**23:
# 				print("out of memory")
# 				return True
			n = self.frontier.pop()
			
			if n.depth >= max_depth:
				return False
			
			self.test_board = self.initialize_test_board(n)
			v+= 1
# 			print('-----------------------------------')
# 			print('f = ', n.f)
# 			print('h = ', n.f- n.depth)
# 			print('d = ',n.depth)
# 			if v % 1000 == 0:
# 				print(v)
# 				self.test_board.pretty_print()
# 				arr = numpy.array(self.test_board.board, dtype = int)
# 				print(arr)
# 				plt.matshow(arr)
 			# self.test_board.pretty_print()
# 			if v == 20:
# 				raise('stop')
# 			print(self.test_board.boxes, self.test_board.storage)
# 			print('-----------------------------------')
			
			moves = copy.deepcopy(self.test_board.get_moves())
# 			self.test_board.pretty_print()
# 			print(moves)
			
			#iterate through the moves to create children for the best current node
			for movement in moves:
				
				
				move = movement[0].copy(),movement[1]
				new_node = Node(n, move, n.depth+1)

				# moves block and returns new block position
				new_box_pos = self.test_board.move_block(move[0],move[1])
				
				board_identifier = self.make_hashable()
				if not board_identifier in self.seen:
					if not self.test_board.isDeadLocked(move[0]):
						# The get_h value is like h(n) ~ heuristic, depth is like g(h) ~ cost
						h = self.get_h()
						if h == 0:
							return True
							# return self.new_node.list_moves()
						new_node.f = h + 0.5 * new_node.depth
						self.frontier.append(new_node)
					

					
				self.test_board.undo_move_block(move[0],move[1])
				
			# sort the frontier based on f
			self.frontier = [i for i,_ in sorted(zip(self.frontier, [-j.f for j in self.frontier]), key = lambda k: k[1])]
# 			print('f order = ', [item.f for item in self.frontier])
		return False
	
	def has_2x2_deadlocks(self, box_coords):
		x,y = box_coords[0], box_coords[1]
		subarr = self.test_board.board[y-1:y+2, x-1:x+2]
# 		print(subarr)
		return grids.check_deadlocks(subarr)
	
	def make_hashable(self):
		output = self.test_board.board
# 		print("dims",len(output), len(output[0]))
		output = output.reshape((len(output) * len(output[0])))
# 		print('after',output)
		return tuple(output)
		
			
	def get_h(self):
		block_positions = self.test_board.boxes
		storage_positions = self.test_board.storage
		h = path_finder.heuristic('m', block_positions, storage_positions, self.moveable_squares_matrix)
		return h
	
	def initialize_test_board(self, node):
		moves = node.list_moves()
		testBoard = copy.deepcopy(self.starting_board)
		for move in moves:
			testBoard.move_block(move[0],move[1])
		return testBoard



	