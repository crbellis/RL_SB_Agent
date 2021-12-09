from environment import Environment

e = Environment(board=[[4, 4, 4, 4, 4, 4, 4, 4],
						[4, 0, 0, 5, 4, 0, 2, 4],
						[4, 0, 0, 0, 1, 3, 0, 4],
						[4, 0, 3, 0, 4, 4, 0, 4],
						[4, 0, 0, 0, 0, 0, 2, 4],
						[4, 4, 4, 4, 4, 4, 4, 4]])
print(e.parseActions())
print(e.saved_deadlocks)
print(e.solvable_subproblems)