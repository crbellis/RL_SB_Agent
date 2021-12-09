from A_Star_Sokoban import A_star
from environment import Environment

if __name__ == "__main__":
	e = Environment(board=[[4, 4, 4, 4, 4, 4, 4, 4,],
			[4, 0, 0, 0, 4, 0, 0, 4,],
			[4, 0, 0, 0, 0, 0, 1, 4,],
			[4, 0, 0, 3, 4, 4, 0, 4,],
			[4, 0, 0, 0, 0, 0, 2, 4,],
			[4, 4, 4, 4, 4, 4, 4, 4,]])
	print(e.board)
	test = A_star(e)
	print(test.solve())

