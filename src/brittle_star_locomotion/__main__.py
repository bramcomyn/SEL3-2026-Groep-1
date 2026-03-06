from biorobot.brittle_star.environment.undirected_locomotion.dual import BrittleStarUndirectedLocomotionEnvironment
from biorobot.brittle_star.environment.undirected_locomotion.shared import BrittleStarUndirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.mjcf.arena.aquarium import AquariumArenaConfiguration, MJCFAquariumArena
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.default import default_brittle_star_morphology_specification

morphology_specification = default_brittle_star_morphology_specification(num_arms=5, num_segments_per_arm=4, use_p_control=True)
arena_configuration = AquariumArenaConfiguration(size=(10, 5))
environment_configuration = BrittleStarUndirectedLocomotionEnvironmentConfiguration(render_mode="rgb_array", simulation_time=5, camera_ids=[1])


def create_environment(morphology_specification, arena_configuration, environment_configuration, backend):
    morphology = MJCFBrittleStarMorphology(morphology_specification)
    arena = MJCFAquariumArena(arena_configuration)
    return BrittleStarUndirectedLocomotionEnvironment.from_morphology_and_arena(
        morphology=morphology, arena=arena, configuration=environment_configuration, backend=backend
    )


env = create_environment(
    morphology_specification=morphology_specification,
    arena_configuration=arena_configuration,
    environment_configuration=environment_configuration,
    backend="MJX",
)
