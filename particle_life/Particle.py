from dataclasses import dataclass

from pygame.math import Vector2


@dataclass
class Particle:
    position: Vector2
    velocity: Vector2
    radius: int
    colour: int
