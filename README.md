# Quantum-Walks-On-NxN-Grids
Code used for the publication: "Scattering Quantum Random Walks on Square Grids and Randomly Generated Mazes" 

arXiv pre-print version: 

The code assmebled in this project was used to generate all of the results presented in the paper mentioned above.  They are python codes that simulate quantum systems, which in principle can be used as algorithms on a quantum computer.  The advantage to studying quantum systems in this way is a more tactile way of 'seeing' under the hood, rather than relying on solely analytical results. 

## Getting Started

All of the code is written for Python.  For an IDE to run the codes, I reccomend using Spyder through the Anaconda Distribution.

link: https://www.anaconda.com/download

The code requires Python 3.5 or higher

Once you have Spyder, or another Python IDE up and running, download all of the python files and put them together in a location somwhere on your computer. For example, store them in a folder on your desktop labeled "Quantum Walks."  It is important that all of the python files be stored in the same location, as they call upon each other frequently to import functions.

For example, many of the codes will call upon the file '-------.py' for functions:

```
import -------- as nt
```
This Github comes with 4 python files that needed to be imported by other codes:

```
1s.py 
```
```
2.py 
```
```
3.py 
```
```
4.py 
```

As a good first test to make sure everything runs properly: open the python file named "----_Test_Code".  If the file runs properly, a messege should print saying that all of the functions imported correctly.

## Classical Simulation of Quantum Systems

When designing new quantum algorithms, often times it is useful to run simulations of the behavior of quantum systems on a clssical computer.  This is precisely what all of these codes do: simulate the results one could expect from running a Quantum Random Walk on NxN Grid graphs.  The advantage of simulating these walks classically is the ability to store information about the state of the system at any given moment.  This allows us to highlight the unique features of these quantum systems, with exact values for state amplitudes, probabilities, etc.  Such a task is in principle impossible with real quantum systems, which is why studying them through classical codes is so insightful.  By studying the "under the hood" properties of these quantum systems, we can better determine whether they have the potential for speedups over classical algorithms. 

## Running The Codes
All of the codes provided in this project run "out of the box" and showcase certain properties of these NxN quantum systems.  Most of the codes produce a plot or print results to the terminal (or both).  For further explination on the results produced by individual codes, a short paragraph is provided at the beginning of each code as well as documentation on each function.  For more information about the overall goal of these codes, I reccomended reading the arXiv paper listed above.

## Coding Style Disclaimer

I am a physicist (quantum computer scientist?), not a professional software engineer.  If my codes fail to meet some coding etiquettes, I apologize!  I've spent quite a good deal of time making the codes as presentable and user-friendly as they are now, but I know they are still a little rough around the edges.


### Contact Me

**Daniel Koch** - dkochsjsu@gmail.com

If you have any questions / interests in the code, or quantum walks in general, feel free to reach out to me.
