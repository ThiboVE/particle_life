import math
from collections.abc import Iterable
from typing import Self

import pygame as pg
from pygame.math import Vector2

from force import ForceSettings, force_func

colourdict = {
    0: "crimson",
    1: "chartreuse3",
    2: "aquamarine3",
    3: "darkorchid2",
    4: "darkgoldenrod1",
    5: "chocolate1",
}


class Particle:
    def __init__(
        self,
        render,
        position: Vector2,
        velocity: Vector2,
        radius: int,
        colour: int,
    ) -> None:
        self.render = render
        self.position: Vector2 = position
        self.velocity: Vector2 = velocity
        self.radius: int = radius
        self.colour: int = colour

    def __repr__(self):
        return f"Particle({self.position = }, self.colour = {colourdict[self.colour]})"

    def __eq__(self, other: Self) -> bool:
        return (
            self.position == other.position
            and self.velocity == other.velocity
            and self.colour == other.colour
        )

    def update(self, particles: Iterable[Self]):
        # update positions
        self.position = self.position + self.velocity * ForceSettings.dt

        # update velocities
        totalForce = Vector2(0, 0)
        for particle in particles:
            if self == particle:
                continue

            d: Vector2 = particle.position - self.position
            r: float = math.hypot(d.x, d.y)

            if 0 < r < ForceSettings.rMax:
                f = force_func(
                    r / ForceSettings.rMax,
                    self.render.matrix[self.colour, particle.colour],
                )
                totalForce += (d / r) * f

        totalForce *= ForceSettings.rMax * ForceSettings.forceFactor
        self.velocity *= ForceSettings.frictionFactor
        self.velocity += totalForce * ForceSettings.dt

        if (
            self.render.WIDTH <= self.position.x * self.render.WIDTH + self.radius
            or self.position.x * self.render.WIDTH - self.radius <= 0
        ):
            self.position.x = abs(self.position.x - 1)
        if (
            self.render.HEIGHT <= self.position.y * self.render.HEIGHT + self.radius
            or self.position.y * self.render.HEIGHT - self.radius <= 0
        ):
            self.position.y = abs(self.position.y - 1)

    def draw(self):
        x, y = self.position.x * self.render.WIDTH, self.position.y * self.render.HEIGHT
        pg.draw.circle(
            self.render.screen, pg.Color(colourdict[self.colour]), (x, y), self.radius
        )
