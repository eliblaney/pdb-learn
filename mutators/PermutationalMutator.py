from mutators.AbstractMutator import AbstractMutator

class PermutationalMutator(AbstractMutator):

    first = True
    lens = None
    pos = None
    self.total = 1

    def __init__(self, model):
        super().__init__("PermutationalMutator", model)
        self.lens = {}
        self.pos = {}
        it = self.options.items()
        for k, v in it:
            l = len(v)
            self.total = self.total * l
            self.lens[k] = l
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

    def has_next(self):
        return self.first or sum(self.pos.values()) > 0

    def get_estimated_total(self):
        return self.total
