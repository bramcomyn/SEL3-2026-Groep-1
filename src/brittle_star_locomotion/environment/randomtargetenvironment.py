from brittle_star_locomotion.environment.environment import Environment

class RandomTargetEnvironment(Environment):
    """Environment where the target position is randomly generated at the start of each episode."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
