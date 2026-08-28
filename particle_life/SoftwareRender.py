import sys

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
        # variables needed to initialize a pygame window
        pg.init()
        self.RES = self.WIDTH, self.HEIGHT = 1200, 720
        self.FPS = 60
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode(self.RES, pg.RESIZABLE)

        self.particle_system = particle_system

    def update(self):
        pg.display.set_caption(str(round(self.clock.get_fps(), 5)))
        pg.display.flip()
        self.clock.tick(self.FPS)

    def draw(self):
        self.screen.fill((10, 0, 10))

        for i in range(len(self.particle_system)):
            particle = self.particle_system[i]
            x, y = particle.position[0] * self.WIDTH, particle.position[1] * self.HEIGHT

            pg.draw.circle(
                self.screen,
                pg.Color(colourdict[particle.colour]),
                (x, y),
                particle.radius,
            )

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                sys.exit()

    def run(self):
        while True:
            self.check_events()
            self.draw()
            self.particle_system.update()

            self.update()
