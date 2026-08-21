import numpy as np
import pygame as pg
from pygame.math import Vector2

from Particle import Particle
from Quadtree import Quadtree
from range import Rectangle


class SoftwareRender:
    def __init__(self):
        # variables needed to initialize a pygame window
        pg.init()
        self.RES = self.WIDTH, self.HEIGHT = 1200, 720
        self.FPS = 60
        self.screen = pg.display.set_mode(self.RES, pg.RESIZABLE)
        self.clock = pg.time.Clock()

        # variables needed for the program
        self.N = 500
        self.m = 6
        self.matrix = np.array(
            [
                [np.random.choice(np.arange(-1, 1, 0.2)) for _ in range(self.m)]
                for _ in range(self.m)
            ]
        )
        self.NODE_CAPACITY = 125
        self.radius = 3

        self.showqtree = False
        self.num_processes = 4
        self.processes = []

        self.particles = np.array(
            [
                Particle(
                    self,
                    position=Vector2(
                        np.random.choice(np.arange(0, 1, 0.01)),
                        np.random.choice(np.arange(0, 1, 0.01)),
                    ),
                    velocity=Vector2(0, 0),
                    radius=self.radius,
                    colour=np.random.choice(np.arange(0, self.m, 1)),
                )
                for _ in range(self.N)
            ]
        )

    def update(self):
        pg.display.set_caption(str(round(self.clock.get_fps(), 5)))
        pg.display.flip()
        self.clock.tick(self.FPS)

    def draw(self):
        self.screen.fill((10, 0, 10))
        for particle in self.particles:
            particle.draw()

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                exit()

            if event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                self.showqtree = not self.showqtree

    def run(self):
        while True:
            self.draw()
            self.check_events()

            boundary = Rectangle(Vector2(0, 0), Vector2(self.WIDTH, self.HEIGHT))
            quadtree = Quadtree(self.NODE_CAPACITY, boundary)

            for particle in self.particles:
                quadtree.insert(particle)

            for idx, particle in enumerate(self.particles):
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

                particle.update(quadtree.particles)

            if self.showqtree:
                quadtree.show(self.screen)

            self.update()


def timer(function):
    from time import perf_counter

    def wrapper(*args, **kwargs):
        before = perf_counter()
        value = function(*args, **kwargs)
        after = perf_counter()
        print(f"'{function.__name__}' took {after - before} seconds to execute!")
        return value

    return wrapper


if __name__ == "__main__":
    app = SoftwareRender()
    app.run()
