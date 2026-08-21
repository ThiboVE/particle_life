import numpy as np
from pygame.math import Vector2

from particle_life import Particle, SoftwareRender


def main() -> None:
    num_particles = 100
    num_colours = 5
    radius = 3

    attraction_matrix = np.array(
        [
            [np.random.choice(np.arange(-1, 1, 0.2)) for _ in range(num_colours)]
            for _ in range(num_colours)
        ]
    )

    particles = [
        Particle(
            position=Vector2(
                np.random.choice(np.arange(0, 1, 0.01)),
                np.random.choice(np.arange(0, 1, 0.01)),
            ),
            velocity=Vector2(0, 0),
            radius=radius,
            colour=np.random.choice(np.arange(0, num_colours, 1)),
        )
        for _ in range(num_particles)
    ]

    app = SoftwareRender(matrix=attraction_matrix, particles=particles)
    app.run()


if __name__ == "__main__":
    main()
