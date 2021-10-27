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
	print(e.__dict__)
	e.pretty_print()
	# print(e.board)
	e.move("d")
	e.pretty_print()
	e.move("r")
	e.pretty_print()
	e.move("u")
	e.pretty_print()
	e.move("l")
	e.pretty_print()

if __name__ == "__main__":
	main()