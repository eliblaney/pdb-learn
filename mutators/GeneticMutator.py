import random
from mutators.AbstractMutator import AbstractMutator

class GeneticMutator(AbstractMutator):

    N = 20
    max_generations = 50
    generation_num = 1
    population = []
    index = 0
    num_elite = 5
    num_worst = 5
    total = 1

    def __init__(self, model, population_size=20, generations=50, num_elite=5, num_worst=5):
        super().__init__("GeneticMutator", model)
        self.N = population_size
        self.max_generations = generations
        self.num_elite = num_elite 
        self.num_worst = num_worst 
        self.total = self.N * self.max_generations

        # Create initial population at random
        for i in range(self.N):
            self.population.append(self.random_model())

    def next(self):
        if not self.hasNext():
            raise Exception("No next model available")

        if self.index == self.N:
            self.index = 0
            self.new_generation()

        next_model = self.population[self.index]
        self.current = next_model.get_options()
        self.index = self.index + 1 
        return next_model

    def new_generation(self):
        self.generation_num = self.generation_num + 1
        
        # sort models in population by fitness ascending
        self.population.sort(key=self.get_fitness, reverse=True)

        # replace the worst models with children based on fitness
        survivors = self.population[self.num_worst:]
        fitnesses = list(map(self.get_fitness, survivors))
        print(fitnesses)
        print(type(fitnesses))
        parents = random.choices(
                survivors,
                weights=fitnesses,
                k=2*self.num_worst
                )
        for i in range(self.num_worst):
            p1 = survivors[i*2]
            p2 = survivors[i*2+1]
            w1 = self.get_fitness(p1)
            w2 = self.get_fitness(p2)
            o = dict(self.options)
            # weighted uniform crossover
            for k, v in o.items():
                o[k] = random.choices([p1, p2], weights=[w1, w2], k=1)[0].get_options()[k]
            # replacing worst models
            survivors[i] = self.model(o)

        # do mutations on remaining non-elite models
        for i in range(self.N - self.num_elite - self.num_worst):
            j = i + self.num_worst
            m = self.population[j]
            o = dict(m.get_options())
            # mutate random property to random value
            k = random.choice(list(o.keys()))
            o[k] = random.choice(self.options[k])
            self.population[j] = self.model(o)

    def has_next(self):
        return self.index < self.N or self.generation_num < self.max_generations

    def get_generation(self):
        return self.generation_num

    def get_fitness(self, model): 
        return model.get_accuracy()

    def random_model(self):
        o = dict(self.options)
        for k in self.options.keys():
            o[k] = random.choice(o[k])

        return self.model(o)

    def get_estimated_total(self):
        return total
