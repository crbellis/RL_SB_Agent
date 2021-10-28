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
	while move != "q":
		e.pretty_print()
		move = input("Enter move: ").lower()
		if move in moveMap.keys():
			e.move(moveMap[move])
if __name__ == "__main__":
	main()