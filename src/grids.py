import itertools
import copy
import numpy as np
import numpy.lib.arraysetops as aso
import pickle




# twos = []
# seq = itertools.product("345", repeat=4)
# for s in seq:
# 	arr = np.fromiter(s, np.byte)
# 	if 3 in arr:
# 		temp = arr.reshape(2,2)
# 		temp = np.array(temp, dtype="bytes")
# 		twos.append(temp)

def make_hashable(grid):
	output = grid.reshape(len(grid[0])*len(grid))
	return tuple(output)

# threes = pickle.load(open("deadlocks_3x3_sans_rotations", "rb"))
# # threes = [np.array([[1,2,3],[4,5,6],[7,8,9]])]
# for i in range(len(threes)):
# 	
# 	identifiers = set()
# 	identifiers.add(make_hashable(threes[i]))
# 	
# 	
# 	rotated90 = np.rot90(threes[i])
# 	if not make_hashable(rotated90) in identifiers:
# 		identifiers.add(make_hashable(rotated90))
# 		threes.append(rotated90)
# 	rotated180 = np.rot90(rotated90)
# 	if not make_hashable(rotated180) in identifiers:
# 		identifiers.add(make_hashable(rotated180))
# 		threes.append(rotated180)
# 	rotated270 = np.rot90(rotated180)
# 	if not make_hashable(rotated270) in identifiers:
# 		identifiers.add(make_hashable(rotated270))
# 		threes.append(rotated270)
# 	flipped = np.fliplr(threes[i])
# 	if not make_hashable(flipped) in identifiers:
# 		identifiers.add(make_hashable(flipped))
# 		threes.append(flipped)
# 	flipped_rotated90 = np.rot90(flipped)
# 	if not make_hashable(flipped_rotated90) in identifiers:
# 		identifiers.add(make_hashable(flipped_rotated90))
# 		threes.append(flipped_rotated90)
# 	flipped_rotated180 = np.rot90(flipped_rotated90)
# 	if not make_hashable(flipped_rotated180) in identifiers:
# 		identifiers.add(make_hashable(flipped_rotated180))
# 		threes.append(flipped_rotated180)
# 	flipped_rotated270 = np.rot90(flipped_rotated180)
# 	if not make_hashable(flipped_rotated270) in identifiers:
# 		identifiers.add(make_hashable(flipped_rotated270))
# 		threes.append(flipped_rotated270)

# temp = set()
# for two in twos:
# 	temp.add(make_hashable(two))
# twos = temp

# new_temp = set()
# for three in threes:
# 	r = np.array(three, dtype="bytes")
# 	new_temp.add(make_hashable(r))


# threes = new_temp

# deadlock_list = [twos, threes]

# pickle.dump(deadlock_list, open("deadlocks", "wb"))




# np.array([[5, 5], [5, 3]])
# np.array([[3, 5], [3, 3]])
# np.array([[3, 5], [4, 3]])
# np.array([[5 4], [4 3]])
# np.array([[5, 5], [3, 3]])
# np.array([[0, 5], [5, 3]])

# 0
# 5 is goal state with a box
# 6 is player in storage


# for i in range(len(deadlock_list)):
# 		rotated = np.rot90(deadlock_list[i])
# 		# make this a function
# 		if (rotated != deadlock_list[i]).any():
# 				deadlock_list.append(rotated)
# 		if (np.rot90(rotated) != deadlock_list[i]).any():
# 				deadlock_list.append(np.rot90(rotated))
# 		if (np.rot90(np.rot90(rotated) != deadlock_list[i])).any():
# 				deadlock_list.append(np.rot90(np.rot90(rotated)))

		# cut array up 2x2 and 3x3s
		# skimage.util.shape.view_as_windows:
		# get a flat version of all arrays
		# check if any element in deadlock list is a complete match ?

		# convert deadlock list to list of byte arrays
		# byte_list = []
		# for i in range(len(deadlock_list)):
		# byte_list.append(deadlock_list[i].tobytes())


# def compare2x2(arr, deadlock):
# 		# turn to a flat list??
# 		i = 0
# 		a = arr.shape[0] - 1
# 		b = arr.shape[1] - 1
# 		while i < a:
# 				j = 0
# 				while j < b:
# 						#temp = np.array([[arr[i][j], arr[i][j + 1]], [arr[i + 1][j], arr[i + 1][j + 1]]])  # all 2x2s
# 						temp = arr[i:i+2, j:j+2]
# 						
# 						if (temp == deadlock).all():
# 								return True
# 						j += 1
# 				i += 1
# 		return False


# def compare3x3(arr, deadlock):
# 		i = 0
# 		a = arr.shape[0] - 2
# 		b = arr.shape[1] - 2
# 		while i < a:
# 				j = 0
# 				while j < b:
# 						temp = arr[i:i+3, j:j+3]
# 						if (temp == deadlock).all():
# 								return True
# 						j += 1
# 				i += 1
# 		return False

def check_deadlocks(arr):
	twos, threes = pickle.load(open("deadlocks", "rb"))
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
		if make_hashable(t) in twos:
			return True
	for thr in three_by_three_sub_arrays:
		if make_hashable(thr) in threes:
			return True
	return False

#old version
# def check_deadlocks(arr):
# 	for k in range(len(deadlock_list)):
# 		if deadlock_list[k].shape[0] == 2:
# 			if compare2x2(arr, deadlock_list[k]):
# 				return True
# 		elif deadlock_list[k].shape[0] == 3:
# 			if compare3x3(arr, deadlock_list[k]):
# 				return True
# 	return False


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


# matplotlib

'''def deadlock_test(grid):
		for j in range(len(deadlock_list)):
				sub = deadlock_list[j]
				if np.all(np.in1d(sub.ravel(), grid.ravel())):
						deadlock_list.append(grid)
						print('deadlock found')
						# add to deadlock list?
						break
				# else:
				# pathfinder to check if all possible pushes lead to deadlock state
'''




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


# threes_prototypes = [
# 								 np.array([[4, 4, 0], [4, 0, 4], [0, 3, 4]]),
# 								 np.array([[4, 4, 0], [4, 0, 4], [0, 3, 3]]),
# 								 np.array([[4, 4, 0], [4, 0, 3], [0, 3, 4]]),
# 								 np.array([[4, 4, 0], [4, 0, 3], [0, 3, 3]]),
# 								 np.array([[4, 3, 0], [4, 0, 3], [0, 3, 3]]),
# 								 np.array([[3, 4, 0], [4, 0, 3], [0, 3, 3]]),
# 								 np.array([[3, 3, 0], [4, 0, 3], [0, 3, 3]]),
# 								 np.array([[4, 3, 0], [3, 0, 3], [0, 3, 3]]),
# 								 np.array([[3, 3, 0], [3, 0, 3], [0, 3, 3]]),
# 								 np.array([[0, 4, 0], [4, 0, 4], [4, 3, 4]]),
# 								 np.array([[0, 4, 0], [4, 0, 4], [4, 3, 3]]),
# 								 np.array([[0, 4, 0], [4, 0, 3], [4, 3, 4]]),
# 								 np.array([[0, 4, 0], [4, 0, 3], [4, 3, 3]]),
# 								 np.array([[0, 4, 0], [4, 0, 4], [3, 3, 3]]),
# 								 np.array([[0, 4, 0], [4, 0, 3], [3, 3, 3]]),
# 								 np.array([[0, 3, 4], [4, 3, 0], [0, 0, 0]]),
# 								 np.array([[0, 4, 0], [3, 0, 3], [3, 3, 3]])]
# def enumerate_special_3x3s(prototype):
# # 	output = []
# 	output_sans_player = []
# 	box_coords = [list(i) for i in list(np.argwhere(prototype == 3))]
# 	empty_coords = [list(i) for i in list(np.argwhere(prototype == 0))]
# 	num_boxes = len(box_coords)
# 	num_empty = len(empty_coords)
# 	
# 	box_seq = list(itertools.product("35", repeat = num_boxes))
# 	empty_seq = list(itertools.product("02", repeat = num_empty))
# 	
# 	for box_list in box_seq:
# 		for empty_list in empty_seq:
# 			item = copy.deepcopy(prototype)
# 			i = 0
# 			for x,y in box_coords:
# 				item[x,y] = int(box_list[i])
# 				i += 1
# 			j = 0
# 			for x,y in empty_coords:
# 				item[x,y] = int(empty_list[j])
# 				j += 1
# 			output_sans_player.append(item)
# 	return output_sans_player
# all_threes = []
# for item in threes_prototypes:
# 	all_threes += enumerate_special_3x3s(item)
