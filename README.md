# Introduction

This is a small passion project trying to implement the Particle Life game in python and learning about new simulation and optimisation concepts along the way. I became interested in this topic after watching some Youtube videos, among which ['Particle Life: simulating "life" with 200000+ particles'](https://youtu.be/SYDAcivFV-U?si=xEQekyYfGvIs_Ikc) inspired me.

# About

Particle life is an artificial life simulation where simple, local rules of attraction and repulsion between different types of coloured particles create complex emergent behaviour that mimics living microorganisms.

<img width="1189" height="670" alt="image" src="https://github.com/user-attachments/assets/d27624fc-bac1-481d-93c8-565865d31192" />


In this project I initially try to implement a simple version of particle life in pygame, and incremently improve the efficiency of the program to simulate more and more particles.

# Optimisations

## CellList algorithm

Particle life is very computationally heavy, as each frame, for each particle, the interaction with all other particles has to be determined. This results in a calculation with an $O(N^2)$ time complexity. As a solution to this bottleneck in particle interaction calculations, I adapted a variation of the [CellList](https://pmc.ncbi.nlm.nih.gov/articles/PMC12910374/) algorithm. This algorithm partitions the simulation space into a fixed number of cells, each with a length equal to the maximum interaction radius $r_{max}$, and simplifies the iteration over particles to an iteration over the cells. For each cell, the interactions of the particles within that cell are only calculated with particles in the same and eight neighbouring cells. This removes the calculation of negligable interactions and significantly increases simulation performance.  


# Quickstart

Clone this repository, and set the current working directory to the root of the project:

`git clone https://github.com/ThiboVE/particle_life.git
cd particle_life`

Run the project: 

`uv run ./main.py`
