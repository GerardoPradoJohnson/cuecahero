"""Small explicit plugin registry; new components register outside the core."""
class Registry:
    def __init__(self):
        self.factories = {}

    def register(self, kind, name, factory):
        key = (kind, name)
        if key in self.factories:
            raise ValueError(f'Duplicate component: {key}')
        self.factories[key] = factory

    def create(self, kind, name, **kwargs):
        try:
            factory = self.factories[(kind, name)]
        except KeyError:
            raise ValueError(f'Unknown {kind}: {name}') from None
        return factory(**kwargs)
