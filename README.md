# Introduction

This is a small passion project trying to implement the Particle Life game in python. This project is inspired by the Youtube video ['Particle Life: simulating "life" with 200000+ particles'](https://youtu.be/SYDAcivFV-U?si=xEQekyYfGvIs_Ikc).

In this project I initially try to implement a simple version of particle life in pygame, and incremently improve the efficiency of the program to simulate more and more particles.

As a solution to the bottleneck of $O(N^2)$ particle interaction calculations, I adapted a variation of the [CellList](https://pmc.ncbi.nlm.nih.gov/articles/PMC12910374/) algorithm.


# Instructions to run the game

This project uses a UV environment to simplify reproducibility.

To run the project: 

`uv run ./main.py`
