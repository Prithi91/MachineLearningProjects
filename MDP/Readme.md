This Project attempts to study reinforcement learning techniques such as Model-based(Value Iteration and Policy Iteration) and
Model-free(Q-learning) approaches. Two Markov Decision Processes were identified for the purpose of this project and they are briefly described in the below sections.

# FrozenLake

Frozenlake is a simple grid problem. The objective is to train an agent to reach the goal state from start state by navigating through a lake(represented as a grid of varying dimensions) that is mostly frozen but has some holes in between. If the agent falls into any of the holes, it receives a reward of 0 and has to start over. On reaching the goal state, it receives a reward of 1. To make the problem more challenging, a measure of randomness was added to the problem. i.e. given a policy and after taking an action as per the policy, the agent will reach its desired state only with probability 0.33 and with probability 0.33 will reach any of its neighboring states other than the desired state. For the purpose of this project, grids of three sizes have been chosen to demonstrate the underpinnings of various RL approaches. The grid sizes are 4x4, 8x8 and 16x16.

# Towers of Hanoi

Towers of Hanoi is a mathematical puzzle consisting of three rods and a number of disks of different sizes, which can slide onto any rod. The puzzle starts with the disks in a neat stack in ascending order of size on one rod, the smallest at the top, thus making a conical shape. The objective of the puzzle is to move the entire stack to another rod, obeying the rules: 1)Only one disk can be moved at a time, 2) Each move consists of taking the upper disk from one of the stacks and placing it on top of another stack or on an empty rod, 3)No larger disk may be placed on top of a smaller disk. The minimal number of moves required to solve a Tower of Hanoi puzzle is 2 n − 1, where n is the number of disks. For the purpose of this assignment, the number of disks to experiment with were chosen as 3,4 and 6 having 27, 81 and 729 states respectively.

#Analysis

A detailed analysis of the performance of various Reinforcement Learning techniques on the above mentioned problems has been provided in Analysis.pdf.
