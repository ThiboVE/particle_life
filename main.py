from particle_life import ParticleSystem, SoftwareRender


def main() -> None:
    particle_system = ParticleSystem.from_setup(n_particles=2000, n_colours=4, seed=11)

    app = SoftwareRender(particle_system=particle_system)
    app.run()


if __name__ == "__main__":
    main()
