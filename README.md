# Introduction

This is a small passion project trying to implement the Particle Life game in python and learning some new concepts along the way. This project is inspired by the Youtube video ['Particle Life: simulating "life" with 200000+ particles'](https://youtu.be/SYDAcivFV-U?si=xEQekyYfGvIs_Ikc).

# About

Particle life is an artificial life simulation where simple, local rules of attraction and repulsion between different types of coloured particles create complex emergent behaviour that mimics living microorganisms.

In this project I initially try to implement a simple version of particle life in pygame, and incremently improve the efficiency of the program to simulate more and more particles.

As a solution to the bottleneck of $O(N^2)$ particle interaction calculations, I adapted a variation of the [CellList](https://pmc.ncbi.nlm.nih.gov/articles/PMC12910374/) algorithm.


# Quickstart

Clone this repository, and set the current working directory to the root of the project:

`git clone https://github.com/ThiboVE/particle_life.git
cd particle_life`

Run the project: 

`uv run ./main.py`
