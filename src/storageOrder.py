from a_star import A_star
from environment import Environment
from copy import deepcopy

def getOrder(environment):
	# loop through storage and remove location and boxes
	temp = deepcopy(environment.board)
	
	strOrder = []
	# environment.storage = [environment.storage[2]] + environment.storage[0:2]
	while(len(strOrder) < len(environment.boxes)):
		for i, storage in enumerate(environment.storage):
			# if storage already in strOrder skip
			# temp env
			tempE = Environment(board=temp)
			# storages to remove from env
			removeStr = deepcopy(tempE.storage)
			removeBoxes = deepcopy(tempE.boxes)

			if storage in strOrder:
				continue
			# remove current storage from storage to remove
			removeStr.remove(storage) 

			# remove solved storage units
			for stor in strOrder:
				tempE.board[stor[1]][stor[0]] = 5
				removeStr.remove(stor)

			# remove remaining storage units
			for stor in removeStr:
				tempE.board[stor[1]][stor[0]] = 0
			# tempE.storage = storage + strOrder

			# remove number of boxes to match storage
			for i in range(len(removeBoxes)-1):
				tempE.board[removeBoxes[0][1]][removeBoxes[0][0]] = 0
				removeBoxes.pop(0)

			tempE.boxes = removeBoxes

			tempE = Environment(board=tempE.to_int())
			a = A_star(tempE)

			if a.solve() == True:
				strOrder.append(storage)
				break
			else:
				if i < len(environment.storage)-1:
					temp = environment.storage[i]
					environment.storage[i] = environment.storage[i+1]
					environment.storage[i+1] = temp
				# otherwise not solvable with current 
				strOrder.pop()

	print("SOLVED ORDER: ", strOrder)
	return strOrder	

		


	


		





