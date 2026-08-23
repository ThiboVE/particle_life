import numpy as np

from particle_life import ParticleSystem, SoftwareRender


def main() -> None:
    num_particles = 500
    num_colours = 5

    positions = np.random.rand(num_particles, 2)
    velocities = np.random.rand(num_particles, 2)
    colours = np.random.randint(0, num_colours, size=(num_particles,))

    attraction_matrix = np.random.uniform(-1, 1, size=(num_colours, num_colours))

    colour_pairs = attraction_matrix[colours[:, None], colours[None, :]]

    particle_system = ParticleSystem(
        positions=positions,
        velocities=velocities,
        colours=colours,
        colour_pairs=colour_pairs,
    )

    app = SoftwareRender(particle_system=particle_system)
    app.run()


if __name__ == "__main__":
    main()
