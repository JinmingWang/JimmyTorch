class DynamicConfig:
    """
    A class to dynamically create instances of a given class with specified arguments.

    Usage:

    custom_cfg = DynamicConfig(CustomClass, arg1=value1, arg2=value2, ...)
    instance = custom_cfg.build()

    # When you want to add or remove arguments:
    custom_cfg.arg3 = value3   # Add an argument
    custom_cfg.remove('arg2')  # Remove an argument
    print(custom_cfg)          # You can print the current configuration
    instance_new = custom_cfg.build()  # Build a new instance with the updated arguments
    """
    def __init__(self, cls: type, **kwargs):
        self.cls = cls
        for k, v in kwargs.items():
            setattr(self, k, v)

    def build(self):
        """
        Build an instance of the class with the current arguments.
        :return: An instance of the class with the current arguments.
        """
        kwargs = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_') and k != "cls"
        }
        return self.cls(**kwargs)

    def add(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def remove(self, key: str) -> None:
        """
        Remove an attribute from the instance.
        :param key: The name of the attribute to remove.
        :return: None
        """
        if hasattr(self, key):
            delattr(self, key)


    def __str__(self):
        args = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_') and k != "cls"
        }
        return f"Config of {self.cls.__name__}: {args}"


    def __repr__(self):
        return self.__str__()