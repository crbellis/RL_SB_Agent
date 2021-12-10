# COMP SCI 271 Sokoban Reinforcement Learning

Authors:
Wyer, Joseph, jwyer, 48643506
Cohen, Nicholas, cohenn1, 79969789
Bellisime, Chase, cbellisi, 16401046

Python version: python 3.9

Note: our environment is set up to fix the mixed up row, col pairs for the boards
with "-" in the name. Meaning, we read the dimension of the boards in those files as columns, rows, instead of rows, columns.

## How to run

<p>
First, to run the program you will need to install the dependencies by running:<br /><br />
<code>pip3 install -r requirements.txt</code><br /><br />
To run the model from main, from the parent directory, you can enter the following command in your terminal:<br /><br />
<code>python3 src/main.py</code><br /><br />
<code>main.py</code> runs all the sokoban.txt files in <code>./benchmarks/</code> by default but this can be configured.<br /><br />
<br />
The output of the above command is the number of moves along with the sequence of moves.
### How to configure

You can run our model by importing <code>test_model</code> from <code>main.py</code>. Or within the <code>main.py</code> file editing the line
<code>files = ["./benchmarks/"+f for f in listdir("./benchmarks/") if isfile(join("./benchmarks/", f)) and "sokoban" in f]</code>
to point to the correct path.

Alternatively, if you import the function, <code>test_model</code> accepts a list as an argument, where the list contains paths to the sokoban.txt file(s).

For example:
<code>files = ["./benchmarks/sokoban01.txt", "./benchmarks/sokoban-02.txt"...]
moves, time = test_model(files)</code>
The function <code>test_model</code> will return a tuple of a list moves and time (in minutes) to execute.

</p>
