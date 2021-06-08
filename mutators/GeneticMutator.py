from mutators.AbstractMutator import AbstractMutator

class GeneticMutator(AbstractMutator):

    lens = None
    pos = None

    def __init__(self, model):
        super().__init__("GeneticMutator", model)
        self.lens = {}
        self.pos = {}
        it = self.options.items()
        for k, v in it:
            self.lens[k] = len(v)
            self.pos[k] = 0

    def next(self):
        if not self.hasNext():
            raise Exception("No next mutation available")
            

        return self.model(self.current)

    def hasNext(self):
        return True
