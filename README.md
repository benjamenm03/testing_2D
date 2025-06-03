# testing_2D

Simple experiments for converting a 1D parallel messaging interface to work in
two dimensions using MPI.

## Building

Compile with an MPI aware C++ compiler:

```bash
mpic++ -std=c++17 main.cpp -o main
```

## Running

Run the resulting binary with `mpirun`, using a square number of processes (for
example four processes):

```bash
mpirun -np 4 ./main
```

