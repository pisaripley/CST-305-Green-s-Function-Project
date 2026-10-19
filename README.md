# CST-305-Green-s-Function-Project
This is the repository for the Green's Function Project in CST-305: Principles of Modeling and Simulation Lecture &amp; Lab instructed by Dr. Ricardo Citro.

***

# CST-305 Project 3: Green's Function and ODE with IVP

**Programmers:** Pisa Ripley and Noah Calderon-Zuniga
**Course:** CST-305: Principles of Modeling and Simulation Lecture & Lab
**Instructor:** Dr. Ricardo Citro
**Institution:** Grand Canyon University

## Overview

This program solves second-order linear ODEs of the form

```
y'' + p(t) y' + q(t) y = g(t),   y(0) = y0,  y'(0) = v0
```

using Green's function. It comes set up with the two assigned problems:

1. `y'' + y = 4`,  `y(0) = y'(0) = 0`
2. `y'' + 4y = t`, `y(0) = y'(0) = 0`

For each ODE the program:

- solves the homogeneous equation with SciPy's `solve_ivp` to get y1 and y2, and plots them
- computes the Wronskian and the Green's function G(t, s)
- computes the Green's function solution y(t)
- checks it against our hand solutions and against SciPy's RK45 solver
- prints a table, plots the solutions, and animates them

## Files

| File | Purpose |
|------|---------|
| `GF1.py` | Main program for the assignment (run this one) |
| `ODEConfig.py` | Holds the ODEs, initial values and hand solutions (imported by the programs, not run directly) |
| `GF2.py` | Optional: network data propagation simulation that goes with the lab question |

All three files must be in the same folder.

## Requirements

- Python 3.10 or newer
- NumPy, SciPy, Matplotlib (and NetworkX for `GF2.py`)

Install the packages with:

```
python -m pip install numpy scipy matplotlib networkx
```

## How to Run

From the project folder, run:

```
python GF1.py
```

1. The program lists the two assigned ODEs and asks `Type in a different ODE instead? (y/n)`.
   - Enter `n` to solve the assigned ODEs.
   - Enter `y` to type your own ODE. Enter p(t), q(t) and g(t) using `t` and Python syntax
     (`*` for multiply, `**` for powers, and `sin`, `cos`, `exp`, `log`, `sqrt`, `pi`), then y(0) and y'(0).
2. The console shows, for each ODE, the Wronskian, a table of values (y1, y2, Green's function solution,
   hand solution and RK45), and the maximum difference between the methods.
3. Three windows open one after another. Close each one to see the next:
   - homogeneous solutions y1 and y2
   - Green's function solution compared with the hand solution and RK45
   - animation of the solutions being drawn

To run the optional network simulation:

```
python GF2.py
```

## Notes

- In PyCharm, turn off **Settings > Tools > Python Plots > Show plots in tool window** so the animation plays in its own window.
- To change the assigned ODEs, edit `ode1` and `ode2` in `ODEConfig.py`.
