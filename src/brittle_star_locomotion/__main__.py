from biorobot.brittle_star.environment.directed_locomotion.shared import (
    BrittleStarDirectedLocomotionEnvironmentConfiguration,
)
from biorobot.brittle_star.mjcf.arena.aquarium import AquariumArenaConfiguration
from biorobot.brittle_star.mjcf.morphology.specification.default import (
    default_brittle_star_morphology_specification,
)
from brittle_star_locomotion.environment.environment import Environment
from brittle_star_locomotion.replay_buffer.replay_buffer import ReplayBuffer
from jax import random

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

key = random.PRNGKey(0)
actions = random.uniform(key, shape=(1,))

# print(env.get_observations())
# for i in range(1000):
#     print(i)
#     env.step(actions)
# print(env.get_observations())

buffer = ReplayBuffer(
    size_observation=env.get_observation_size(),
    size_action=1)

for i in range(10):
    buffer.add(
        observation=env.get_observations()[0],
        action=actions,
        reward=1.0,
        next_observation=env.get_observations()[0],
        done=False
    )

print(buffer.sample(5))
