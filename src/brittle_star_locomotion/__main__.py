from biorobot.brittle_star.environment.directed_locomotion.shared import BrittleStarDirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.mjcf.arena.aquarium import AquariumArenaConfiguration
from biorobot.brittle_star.mjcf.morphology.specification.default import default_brittle_star_morphology_specification

from brittle_star_locomotion.environment.environment import Environment

morphology_specification = default_brittle_star_morphology_specification(num_arms=5, num_segments_per_arm=4, use_p_control=True)
arena_configuration = AquariumArenaConfiguration(size=(10, 5), attach_target=True)
environment_configuration = BrittleStarDirectedLocomotionEnvironmentConfiguration(render_mode="rgb_array", simulation_time=5, camera_ids=[1])


env = Environment(
    5,
    3,
    arena_configuration=arena_configuration,
    environment_configuration=environment_configuration,
    observations=["actuator_force", "joint_position"],
)
