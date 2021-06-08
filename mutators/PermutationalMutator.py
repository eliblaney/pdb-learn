from mutators.AbstractMutator import AbstractMutator

class PermutationalMutator(AbstractMutator):

    first = True
    lens = None
    pos = None

    def __init__(self, model):
        super().__init__("PermutationalMutator", model)
        self.lens = {}
        self.pos = {}
        it = self.options.items()
        for k, v in it:
            self.lens[k] = len(v)
            self.pos[k] = 0

    def next(self):
        if not self.hasNext():
            raise Exception("No next mutation available")
            
        self.first = False
        self.current = {}
        for k, v in self.pos.items():
            self.current[k] = self.options[k][v]
        
        for k, v in self.pos.items():
            self.pos[k] = (v + 1) % self.lens[k]
            if self.pos[k] > 0:
                break

        return self.model(self.current)

    def hasNext(self):
        return self.first or sum(self.pos.values()) > 0
