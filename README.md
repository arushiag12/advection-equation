# Advection Equation Evolution Using Distributed-Memory Parallel Lax

## Project Overview

This project implements a distributed-memory parallel Lax-Wendro scheme to numerically solve the advection equation. The parallelization is achieved using MPI and OpenMP, adhering to the specified initial condition outlined in the project writeup.

## Code Structure

In the initialization phase, appropriate values are assigned to squares in each process based on the initial condition. During the update for each time step, when the code is executed with more than one MPI process, it employs a variant of the 'checkerboard' approach for sending and receiving across processes. Specifically, the checkerboard approach is used for trading the internal ghost cells (excluding those on the boundary) of the final matrix. Then the code exchanges right ghost cells with the left, and vice versa, while similar exchanges occur for the up-down cells.

## Execution Parameters
Execution time measurements are conducted with N = 4000.
Plots are generated with N = 1000.
Other parameters utilized are:
- L = 1.0
- T = 1.0
- u = sqrt(2) y
- v = - sqrt(2) x
- del_t = 1.25e-4

