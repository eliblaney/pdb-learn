
class AbstractMutator:

    name = "UnknownMutator"
    model = None
    options = {}
    current = None

    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.options = model.get_default_options()

    def get_name(self):
        return self.name

    def get_current_options(self):
        return self.current

    def next(self):
        raise NotImplementedError("Subclass must implement next()")

    def hasNext(self):
        raise NotImplementedError("Subclass must implement hasNext()")

    def __str__(self):
        return self.name
