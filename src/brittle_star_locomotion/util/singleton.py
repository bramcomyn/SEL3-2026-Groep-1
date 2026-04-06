class Singleton(type):
    """Singleton baseclass. Any class has `Singleton` as a metaclass will be a singleton.

    Use case:
    >>> class MySingleton(metaclass=Singleton):
    >>> def __init__(self, value):
    >>>     self.value = value

    >>> instance1 = MySingleton(10)
    >>> instance2 = MySingleton(20)

    >>> print(instance1.value)  # Output: 10
    >>> print(instance2.value)  # Output: 10
    >>> print(instance1 is instance2)  # Output: True
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
