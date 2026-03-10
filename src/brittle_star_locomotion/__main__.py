from typing import Literal

import jax
from biorobot.brittle_star.environment.directed_locomotion.dual import (
    BrittleStarDirectedLocomotionEnvironment,
)
from biorobot.brittle_star.environment.directed_locomotion.shared import (
    BrittleStarDirectedLocomotionEnvironmentConfiguration,
)
from biorobot.brittle_star.environment.light_escape.dual import (
    BrittleStarLightEscapeEnvironment,
)
from biorobot.brittle_star.environment.light_escape.shared import (
    BrittleStarLightEscapeEnvironmentConfiguration,
)
from biorobot.brittle_star.environment.undirected_locomotion.dual import (
    BrittleStarUndirectedLocomotionEnvironment,
)
from biorobot.brittle_star.environment.undirected_locomotion.shared import (
    BrittleStarUndirectedLocomotionEnvironmentConfiguration,
)
from biorobot.brittle_star.mjcf.arena.aquarium import (
    AquariumArenaConfiguration,
    MJCFAquariumArena,
)
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.default import (
    default_brittle_star_morphology_specification,
)
from biorobot.brittle_star.mjcf.morphology.specification.specification import (
    BrittleStarMorphologySpecification,
)
from moojoco.environment.dual import DualMuJoCoEnvironment

seed = 42

morphology_specification = default_brittle_star_morphology_specification(
    num_arms=5,
    num_segments_per_arm=4,
    # use position-based control
    use_p_control=True,
    # do not use torque-based control
    use_torque_control=False,
)

def create_morphology(
    morphology_specification: BrittleStarMorphologySpecification
) -> MJCFBrittleStarMorphology:
    morphology = MJCFBrittleStarMorphology(specification=morphology_specification)
    return morphology

arena_configuration = AquariumArenaConfiguration(
    size=(10, 5),
    sand_ground_color=True,
    attach_target=True,
    wall_height=1.5,
    wall_thickness=0.1
)

def create_arena(
    arena_configuration: AquariumArenaConfiguration
) -> MJCFAquariumArena:
    arena = MJCFAquariumArena(configuration=arena_configuration)
    return arena

shared_configuration = {
    "joint_randomization_noise_scale": 0.0,
    "render_mode": "rgb_array",
    "simulation_time": 5,
    "num_physics_steps_per_control_step": 10,
    "time_scale": 2,
    "camera_ids": (0, 1),
    "render_size": (480, 640)
}

locomotion_env_configuration = BrittleStarUndirectedLocomotionEnvironmentConfiguration(
    **shared_configuration
)

target_locomotion_env_configuration = \
    BrittleStarDirectedLocomotionEnvironmentConfiguration(
    target_distance=3.0,
    **shared_configuration
)

light_escape_env_configuration = \
    BrittleStarLightEscapeEnvironmentConfiguration(
    light_perlin_noise_scale=0,
    **shared_configuration
)

env_config_type = BrittleStarDirectedLocomotionEnvironmentConfiguration | \
                  BrittleStarUndirectedLocomotionEnvironmentConfiguration | \
                  BrittleStarLightEscapeEnvironmentConfiguration

backend_type = Literal["MJC", "MJX"]

def create_environment(
    morphology_specification: BrittleStarMorphologySpecification,
    arena_configuration: AquariumArenaConfiguration,
    environment_configuration: env_config_type,
    backend: backend_type
) -> DualMuJoCoEnvironment:
    morphology = create_morphology(morphology_specification=morphology_specification)
    arena      = create_arena(arena_configuration=arena_configuration)

    if isinstance(
        environment_configuration,
        BrittleStarUndirectedLocomotionEnvironmentConfiguration
    ):
        env_class = BrittleStarUndirectedLocomotionEnvironment
    elif isinstance(
        environment_configuration,
        BrittleStarDirectedLocomotionEnvironmentConfiguration
    ):
        env_class = BrittleStarDirectedLocomotionEnvironment
    elif isinstance(
        environment_configuration,
        BrittleStarLightEscapeEnvironmentConfiguration
    ):
        env_class = BrittleStarLightEscapeEnvironment
    else:
        return

    environment = env_class.from_morphology_and_arena(
        morphology=morphology,
        arena=arena,
        configuration=environment_configuration,
        backend=backend
    )

    return environment

if __name__ == "__main__":
    morphology = create_morphology(morphology_specification)
    arena      = create_arena(arena_configuration)
    environment = create_environment(
        morphology_specification=morphology_specification,
        arena_configuration=arena_configuration,
        environment_configuration=target_locomotion_env_configuration,
        backend="MJX"
    )

    rng = jax.random.PRNGKey(seed=seed)
    env_rng, action_rng = jax.random.split(rng, 2)

    jit_step  = jax.jit(environment.step)
    jit_reset = jax.jit(environment.reset)

    state = jit_reset(rng=env_rng)
