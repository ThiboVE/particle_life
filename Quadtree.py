import pygame as pg
from pygame.math import Vector2
from range import Rectangle
from Particle import Particle


class Quadtree:
    def __init__(self, capacity: int, boundary: Rectangle):
        self.capacity: int = capacity
        self.boundary: Rectangle = boundary
        self.particles = []
        self.color = pg.Color("White")

        self.divided = False

    def subdivide(self):
        parent = self.boundary

        boundary_nw = Rectangle(Vector2(parent.position.x, parent.position.y), parent.scale / 2)

        boundary_ne = Rectangle(Vector2(parent.position.x + parent.scale.x / 2, parent.position.y), parent.scale / 2)

        boundary_sw = Rectangle(Vector2(parent.position.x, parent.position.y + parent.scale.y / 2), parent.scale / 2)

        boundary_se = Rectangle(Vector2(parent.position.x + parent.scale.x / 2, parent.position.y + parent.scale.y / 2), parent.scale / 2)

        self.northWest = Quadtree(self.capacity, boundary_nw)
        self.northEast = Quadtree(self.capacity, boundary_ne)
        self.southWest = Quadtree(self.capacity, boundary_sw)
        self.southEast = Quadtree(self.capacity, boundary_se)

        self.divided = True

    def insert(self, particle: Particle) -> bool:
        if not self.boundary.containsParticle(particle):
            return False

        if len(self.particles) < self.capacity:
            self.particles.append(particle)
            return True

        if not self.divided:
            self.subdivide()

        return (self.northWest.insert(particle) or
                self.northEast.insert(particle) or
                self.southWest.insert(particle) or
                self.southEast.insert(particle))

    def queryRange(self, _range: Rectangle, found: list = None):
        if found is None:
            found = []

        if not _range.intersects(self.boundary):
            return False

        for particle in self.particles:
            if _range.containsParticle(particle):
                found.append(particle)

        if self.divided:
            self.northWest.queryRange(_range, found)
            self.northEast.queryRange(_range, found)
            self.southWest.queryRange(_range, found)
            self.southEast.queryRange(_range, found)

        return found

    def show(self, screen):
        self.boundary.draw(screen)
        if self.divided:
            self.northWest.show(screen)
            self.northEast.show(screen)
            self.southWest.show(screen)
            self.southEast.show(screen)
