import sys
import time

import pygame as pg

from particle_life.Particle import ParticleSystem

colourdict = {
    0: "crimson",
    1: "chartreuse3",
    2: "aquamarine3",
    3: "darkorchid2",
    4: "darkgoldenrod1",
    5: "chocolate1",
}


class SoftwareRender:
    def __init__(self, particle_system: ParticleSystem) -> None:
        pg.init()
        self.RES = self.WIDTH, self.HEIGHT = 1200, 720
        self.FPS = 60
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode(self.RES, pg.RESIZABLE)

        self.particle_system = particle_system

    def update(self) -> None:
        pg.display.set_caption(str(round(self.clock.get_fps(), 5)))
        pg.display.flip()
        self.clock.tick(self.FPS)

    def draw(self) -> None:
        self.screen.fill((0, 0, 0))

        for particle in self.particle_system:
            x, y = particle.position[0] * self.WIDTH, particle.position[1] * self.HEIGHT

            pg.draw.circle(
                self.screen,
                pg.Color(colourdict[particle.colour]),
                (x, y),
                particle.radius,
            )

    def check_events(self) -> None:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                sys.exit()

            if event.type == pg.VIDEORESIZE:
                self.WIDTH = event.w
                self.HEIGHT = event.h

    def run(self) -> None:
        while True:
            self.check_events()
            self.update()

            t0 = time.perf_counter()
            self.draw()
            t1 = time.perf_counter()

            self.particle_system.update()
            t2 = time.perf_counter()

            # print(f"drawing: {(t1 - t0):2f}")
            # print(f"physics: {(t2 - t1):2f}")
