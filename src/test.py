from environment import Environment

e = Environment("./benchmarks/sokoban-02.txt")
print(e.saved_deadlocks)
print(e.solvable_subproblems)