from brittle_star_locomotion.util.singleton import Singleton


class ExampleSingleton(metaclass=Singleton):
    init_calls = 0

    def __init__(self, value):
        ExampleSingleton.init_calls += 1
        self.value = value


def teardown_function(_function):
    """Reset singleton cache between tests to avoid order-dependent state leakage."""
    Singleton._instances.pop(ExampleSingleton, None)
    ExampleSingleton.init_calls = 0


def test_singleton_returns_same_instance():
    """Repeated construction should return the same object identity."""
    instance_a = ExampleSingleton(10)
    instance_b = ExampleSingleton(20)

    assert instance_a is instance_b


def test_singleton_initializes_only_once():
    """Only first construction should run __init__, preserving the first assigned value."""
    instance_a = ExampleSingleton(10)
    instance_b = ExampleSingleton(20)

    assert ExampleSingleton.init_calls == 1
    assert instance_a.value == 10
    assert instance_b.value == 10
