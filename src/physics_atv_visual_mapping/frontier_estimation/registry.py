class Registry(dict):
    def register(self, name):
        def decorator(cls):
            self[name] = cls
            return cls
        return decorator

    def build(self, name, **kwargs):
        if name not in self:
            raise KeyError(f"{name} not in registry. Available: {list(self.keys())}")
        return self[name](**kwargs)
