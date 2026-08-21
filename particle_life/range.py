import pygame as pg
from pygame.math import Vector2


class Rectangle:
    def __init__(self, position: Vector2, scale: Vector2):
        self.position = position
        self.scale = scale
        self.color = (255, 255, 255)
        self.lineThickness = 1

    def containsParticle(self, particle) -> bool:
        bx, by = self.position
        x, y = particle.position
        x, y = x * 1200, y * 720  # scale up the positions to screen
        w, h = self.scale
        return bx <= x <= bx + w and by <= y <= by + h

    def intersects(self, _range) -> bool:
        x, y = self.position
        w, h = self.scale
        xr, yr = _range.position
        wr, hr = _range.scale
        return not (
            xr - wr > x + w or xr + wr < x - w or yr - hr > y + h or yr + hr < y - h
        )

    def draw(self, screen):
        x, y = self.position
        w, h = self.scale
        pg.draw.rect(screen, self.color, [x, y, w, h], self.lineThickness)


class Circle:
    def __init__(self, position: Vector2, radius: int):
        self.position = position
        self.radius = radius
        self.sqradius = self.radius * self.radius
        self.scale = None
        self.color = (255, 255, 255)
        self.lineThickness = 1

    def containsParticle(self, particle) -> bool:
        x1, y1 = self.position
        x2, y2 = particle.position
        dist = pow(x2 * 1200 - x1, 2) + pow(y2 * 720 - y1, 2)
        return dist <= self.sqradius / 1399.428455

    def intersects(self, _range) -> bool:
        x1, y1 = self.position
        x2, y2 = _range.position
        w, h = _range.scale
        r = self.radius
        dist_x, dist_y = abs(x2 - x1), abs(y2 - y1)

        edges = pow(dist_x - w, 2) + pow(dist_y - h, 2)

        if dist_x > (r + w) or dist_y > (r + h):
            return False

        if dist_x <= w or dist_y <= h:
            return True

        return edges <= self.sqradius

    def draw(self, screen):
        pg.draw.circle(
            screen, self.color, self.position, self.radius, self.lineThickness
        )
