import math

import numpy as np
import pygame as pg
from pygame.math import Vector2

from particle_life.force import ForceSettings, force_func
from particle_life.Particle import Particle
from particle_life.Quadtree import Quadtree
from particle_life.range import Rectangle

colourdict = {
    0: "crimson",
    1: "chartreuse3",
    2: "aquamarine3",
    3: "darkorchid2",
    4: "darkgoldenrod1",
    5: "chocolate1",
}


class SoftwareRender:
    def __init__(self, matrix: np.ndarray, particles: list[Particle]) -> None:
        # variables needed to initialize a pygame window
        pg.init()
        self.RES = self.WIDTH, self.HEIGHT = 1200, 720
        self.FPS = 60
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode(self.RES, pg.RESIZABLE)

        self.particles = particles
        self.matrix = matrix

        self.NODE_CAPACITY = 125
        self.showqtree = False
        self.num_processes = 4
        self.processes = []

    def update(self):
        pg.display.set_caption(str(round(self.clock.get_fps(), 5)))
        pg.display.flip()
        self.clock.tick(self.FPS)

    def draw(self):
        self.screen.fill((10, 0, 10))
        for particle in self.particles:
            x, y = particle.position.x * self.WIDTH, particle.position.y * self.HEIGHT
            pg.draw.circle(
                self.screen,
                pg.Color(colourdict[particle.colour]),
                (x, y),
                particle.radius,
            )

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                exit()

            if event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                self.showqtree = not self.showqtree

    def update_particle(self, particle: Particle, particles: list[Particle]) -> None:
        # update positions
        particle.position = particle.position + particle.velocity * ForceSettings.dt

        # update velocities
        totalForce = Vector2(0, 0)
        for particle2 in particles:
            if particle == particle2:
                continue

            d: Vector2 = particle2.position - particle.position
            r: float = math.hypot(d.x, d.y)

            if 0 < r < ForceSettings.rMax:
                f = force_func(
                    r / ForceSettings.rMax,
                    self.matrix[particle.colour, particle2.colour],
                )
                totalForce += (d / r) * f

        totalForce *= ForceSettings.rMax * ForceSettings.forceFactor
        particle.velocity *= ForceSettings.frictionFactor
        particle.velocity += totalForce * ForceSettings.dt

        if (
            self.WIDTH <= particle.position.x * self.WIDTH + particle.radius
            or particle.position.x * self.WIDTH - particle.radius <= 0
        ):
            particle.position.x = abs(particle.position.x - 1)
        if (
            self.HEIGHT <= particle.position.y * self.HEIGHT + particle.radius
            or particle.position.y * self.HEIGHT - particle.radius <= 0
        ):
            particle.position.y = abs(particle.position.y - 1)

    def run(self):
        while True:
            self.draw()
            self.check_events()

            boundary = Rectangle(Vector2(0, 0), Vector2(self.WIDTH, self.HEIGHT))
            quadtree = Quadtree(self.NODE_CAPACITY, boundary)

            for particle in self.particles:
                quadtree.insert(particle)

            for particle in self.particles:
                self.update_particle(particle, quadtree.particles)
                """range_scale = Vector2(200, 100)
                range_position = Vector2((particle.position.x * self.WIDTH) - (0.5 * range_scale.x), (particle.position.y * self.HEIGHT) - (0.5 * range_scale.y))
                range = Rectangle(range_position, range_scale)
    
                # others = quadtree.queryRange(range)
    
                if idx == 0:
                    range.draw(self.screen)
                    # print(others)
                

                print(particle)
                print(quadtree.particles)
                print("\n")
                print("\n")"""

            #     particle.update(quadtree.particles)

            # if self.showqtree:
            #     quadtree.show(self.screen)

            self.update()
