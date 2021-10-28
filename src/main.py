from environment import Environment
def main():
	"""
	format of input file:
	line0 = sizeH sizeV
	line1 = number of boxes and coordinates
	line2 = number of boxes and coordinates
	line3 = number of storage locations and coordinates
	line4 = player's starting coordinates =
	"""
	file = "./input/sokoban01.txt"
	e = Environment()
	e.read_file(file)
	moveMap = {"w": "u", "d": "r", "a": "l", "s": "d"}
	move = ""
	moves = []
	while move != "q" and not e.terminal():
		e.pretty_print()
		move = input("Enter move (q to quit): ").lower()
		if move in moveMap.keys():
			moves.append(move)
			e.move(moveMap[move])
	if e.terminal():
		e.pretty_print()
		print("You won! Here were your moves: ", moves)
	else:
		print("you quit")

if __name__ == "__main__":
	main()
