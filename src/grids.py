import itertools
import numpy as np

# Makes numpy array hashable
def make_hashable(grid):
	output = grid.reshape(len(grid[0])*len(grid))
	return tuple(output)


# Uesd to help identify deadlocks to save
def a_star_test(arr):  # create padded grid for a-star
		"""
		size = arr.shape
		rows = size[0]
		columns = size[1]
		if rows < 3 or columns < 3:
				print("2x2 deadlocks already found ")
				return 1"""

		count_boxes = np.count_nonzero(arr == b'3')  # of boxes on grid
		count_goals = np.count_nonzero(arr == b'2') + np.count_nonzero(arr == b'6')  # of goals on grid
		# need to count boxes already on goal? (arr == 5)
		new_array = np.pad(arr, 4, mode='constant', constant_values=b'0')  # add 4 empty spaces around grid
		new_array = np.pad(new_array, 1, 'constant', constant_values=b'4')

		if count_boxes > count_goals:  # more boxes than goals
				diff = count_boxes - count_goals
				if diff <= new_array.shape[1] - 2:
						for i in range(1, diff + 1):
								new_array[1, i] = 2  # add goals to first row of padded grid starting at (0,0)
				else:
						for i in range(1, new_array.shape[1] - 1):
								new_array[1, i] = 2
						for i in range(new_array.shape[1] - 1, diff + 1):
								new_array[new_array.shape[0] - 2, i - (new_array.shape[0] - 3)] = 2
		elif count_goals > count_boxes:  # more goals than boxes
				diff = count_goals - count_boxes
				if diff <= new_array.shape[1] - 3:
						for i in range(2, diff + 2):
								new_array[2, i] = 3  # add boxes starting at (1, 1) so player can reach them
				else:
						for i in range(2, new_array.shape[1] - 2):
								new_array[2, i] = 3
						for i in range(new_array.shape[1] - 2, diff + 2):
								new_array[new_array.shape[0] - 3, i - (new_array.shape[0] - 4)] = 3
		if not b'1' in arr and not b'6' in arr:
			if new_array[1,1] == b'2':
				new_array[1,1] = b'6'
			elif new_array[1,1] == b'0':
				new_array[1,1] = b'1'
			else:
				print(new_array[1,1], type(new_array[1,1]))
				raise('something went wrong')
		return new_array





# Used to help identify deadlocks to save"
def enumerate_grids(n):
	grids = []
	n_square = n * n
	seq = itertools.product("02345", repeat=n_square)
	for s in seq:
		# next(seq)
		arr = np.fromiter(s, np.int8)
		grids.append(arr.reshape(n,n))
	player = list(itertools.product("16", repeat=1))
	with_player = n * n -1
	seq = list(itertools.product("02345", repeat=with_player))
	for s in seq:
		for i in range(n*n):
			first_half = s[0:i]
			second_half = s[i:]
			for j in player:
				composite = first_half + j + second_half
				arr = np.fromiter(composite, np.int8)
				grids.append(arr.reshape(n,n))
	return grids
