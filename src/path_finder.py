import queue, numpy
from scipy.optimize import linear_sum_assignment
# from sklearn import preprocessing

# Run test() to see how it works. The matplotlib stuff visualizes the matrix for you.
# Not totally necessary to install to test it.

import matplotlib.pyplot as plt
def test():
	matrix = numpy.array([[False for i in range(10)] for j in range(10)])
	plt.matshow(matrix)
	for i in range(10):
		matrix[i][0] = True
		matrix[9][i] = True
	for i in range(5):
		matrix[9-i][9] = True
		matrix[5][9-i] = True
	plt.matshow(matrix)
	pos1 = (0,0)
	pos2 = (5,5)
	path(pos1, pos2, 100, matrix)
	

# type = 'm' for manhattan distance, otherwise uses maze distance
# takes list of coords of block positions and storage positions
# takes a boolean matrix True=walkable, for moveable_squares_matrix
# returns an admissible heuristic value
def heuristic(heuristic_type, block_positions, storage_positions, moveable_squares_matrix=None):
	"""
	returns the least amount of movements to move boxes to all storage units
	"""
	if heuristic_type == 'm':
		costs = numpy.array([[manhattan_distance(block_positions[i], storage_positions[j]) for i in range(len(block_positions))] for j in range(len(storage_positions))])
	else:
		costs = numpy.array([[maze_search(block_positions[i], storage_positions[j]) for i in range(len(block_positions))] for j in range(len(storage_positions))])
	row_indices, collumn_indices = linear_sum_assignment(costs)
	costs = numpy.array(costs, "float64")
	return costs.max() 

def manhattan_distance(pos1, pos2):
	return abs(pos1[0] - pos2[0]) + abs(pos1[1]-pos2[1])

def maze_search(pos1, pos2, max_distance, matrix):
	return maze_search_tree(pos1,pos2,max_distance,matrix).d

def path(pos1, pos2, max_distance, matrix):
	currentNode = maze_search_tree(pos1,pos2, max_distance, matrix).solution
	output = [[] for i in range(currentNode.depth)]
	for i in range(1,currentNode.depth+1):
		output[-i] = currentNode.pos
		currentNode = currentNode.parent
	#print(output)
	return output
		
	
	


# matrix is a matrix of Trues and Falses. Trues are accessible walkways, and Falses are not.
class maze_search_tree: 
	def __init__(self, pos1, pos2, max_distance, matrix):
		self.matrix = numpy.matrix(matrix)
		self.pos1 = pos1
		self.pos2 = pos2
		self.root = maze_search_node(pos1, 0, None, [])
		self.frontier = queue.Queue(len(matrix)*len(matrix[0]))
		self.frontier.put(self.root)
		self.solution = self.maze_path(max_distance)
		self.d = self.solution.depth
	
	
	def maze_path(self, max_distance):
		self.matrix[self.pos1] = False
		while not self.frontier.empty():
			expanding = self.frontier.get()
			pos = expanding.pos
			
			#down
			down_pos = pos[0]+1,pos[1]
			if down_pos == self.pos2:
				destination_node = maze_search_node(down_pos, expanding.depth + 1, expanding, [])
				expanding.children += [destination_node]
				return destination_node
			if down_pos[0] < len(self.matrix) and self.matrix[down_pos]:
				expanding.children += [maze_search_node(down_pos, expanding.depth + 1, expanding, [])]
				self.matrix[down_pos] = False
				
			#up
			up_pos = pos[0]-1,pos[1]
			if up_pos == self.pos2:
				destination_node = maze_search_node(up_pos, expanding.depth + 1, expanding, [])
				expanding.children += [destination_node]
				return destination_node
			if up_pos[0] > 0 and self.matrix[up_pos]:
				expanding.children += [maze_search_node(up_pos, expanding.depth + 1, expanding, [])]
				self.matrix[up_pos] = False 
			
			#left
			left_pos = pos[0],pos[1]-1
			if left_pos == self.pos2:
				destination_node = maze_search_node(left_pos, expanding.depth + 1, expanding, [])
				expanding.children += [destination_node]
				return destination_node
			if left_pos[1] > 0 and self.matrix[left_pos]:
				expanding.children += [maze_search_node(left_pos, expanding.depth + 1, expanding, [])]
				self.matrix[left_pos] = False 
			
			#right
			right_pos = pos[0],pos[1]+1
			if right_pos == self.pos2:
				destination_node = maze_search_node(right_pos, expanding.depth + 1, expanding, [])
				expanding.children += [destination_node]
				return destination_node
			if pos == (9,0):
				print(right_pos, right_pos[1] < len(self.matrix[:,0]), self.matrix[right_pos])
			if right_pos[1] < len(self.matrix[:,0]) and self.matrix[right_pos]:
				expanding.children += [maze_search_node(right_pos, expanding.depth + 1, expanding, [])]
				self.matrix[right_pos] = False 
			
			# print(expanding.children)
			for child in expanding.children:
				self.frontier.put(child)
				
			if expanding.depth > max_distance:
				return None
			
		return None
	
	def maze_distance(self):
		if not self.matrix[self.pos1] or not self.matrix[self.pos2]:
			return -1
		self.matrix[self.pos1] = False
		while not self.frontier.empty():
			expanding = self.frontier.get()
			pos = expanding.pos
			# print(pos)
			# plt.matshow(self.matrix)
			
			#down
			down_pos = pos[0]+1,pos[1]
			if down_pos == self.pos2:
				return expanding.depth + 1
			if down_pos[0] < len(self.matrix) and self.matrix[down_pos]:
				expanding.children += [maze_search_node(down_pos, expanding.depth + 1, expanding, [])]
				self.matrix[down_pos] = False
				
			#up
			up_pos = pos[0]-1,pos[1]
			if up_pos == self.pos2:
				return expanding.depth + 1
			if up_pos[0] > 0 and self.matrix[up_pos]:
				expanding.children += [maze_search_node(up_pos, expanding.depth + 1, expanding, [])]
				self.matrix[up_pos] = False 
			
			#left
			left_pos = pos[0],pos[1]-1
			if left_pos == self.pos2:
				return expanding.depth + 1
			if left_pos[1] > 0 and self.matrix[left_pos]:
				expanding.children += [maze_search_node(left_pos, expanding.depth + 1, expanding, [])]
				self.matrix[left_pos] = False 
			
			#right
			right_pos = pos[0],pos[1]+1
			if right_pos == self.pos2:
				return expanding.depth + 1
			if pos == (9,0):
				print(right_pos, right_pos[1] < len(self.matrix[:,0]), self.matrix[right_pos])
			if right_pos[1] < len(self.matrix[:,0]) and self.matrix[right_pos]:
				expanding.children += [maze_search_node(right_pos, expanding.depth + 1, expanding, [])]
				self.matrix[right_pos] = False 
			
			# print(expanding.children)
			for child in expanding.children:
				self.frontier.put(child)

class maze_search_node:
	
	def __init__(self, pos, depth, parent, children):
		self.pos = pos
		self.depth = depth
		self.parent = parent
		self.children = children
	def isRoot(self):
		if self.parent == None:
			return True
		else:
			return False