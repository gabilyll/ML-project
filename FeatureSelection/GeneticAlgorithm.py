import random
import time

import numpy as np
from deap import base, creator, tools
from sklearn.metrics import accuracy_score

import test
from DCA import DCA


def calculate_fitness(model, x, y):
    print('--------------------------------------------')
    print(x.shape)
    model.fit(x, y)

    predicted_y = model.predict(y)

    res = accuracy_score(y, predicted_y)
    print('acc: ', res)

    return res


class FeatureSelectionGA:

    def __init__(self, model, x, y, verbose=0):
        self.model = model
        self.n_features = x.shape[1]
        self.toolbox = None
        self.creator = self._create()
        # self.cv_split = cv_split
        self.x = x
        self.y = y
        self.verbose = verbose
        if self.verbose == 1:
            print(
                "Model {} will select best features among {} features.".format(
                    model, x.shape[1]
                )
            )
            print("Shape od train_x: {} and target: {}".format(x.shape, y.shape))
        self.final_fitness = []
        self.fitness_in_generation = {}
        self.best_ind = None

    def evaluate(self, individual):
        np_ind = np.asarray(individual)
        if np.sum(np_ind) == 0:
            fitness = 0.0
        else:
            feature_idx = np.where(np_ind == 1)[0]
            print('feature_idx: ', feature_idx)
            fitness = calculate_fitness(
                self.model, self.x[:, feature_idx], self.y
            )

        if self.verbose == 1:
            print("Individual: {}  Fitness_score: {} ".format(individual, fitness))

        return (fitness,)

    def _create(self):
        creator.create("FeatureSelect", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FeatureSelect)
        return creator

    def create_toolbox(self):
        """
        Custom creation of toolbox.
        Parameters
        -----------
            self
        Returns
        --------
            Initialized toolbox
        """

        self._init_toolbox()
        return self.toolbox

    def register_toolbox(self, toolbox):
        """
        Register custom created toolbox. Evalute function will be registerd
        in this method.
        Parameters
        -----------
            Registered toolbox with crossover,mutate,select tools except evaluate
        Returns
        --------
            self
        """
        toolbox.register("evaluate", self.evaluate)
        self.toolbox = toolbox

    def _init_toolbox(self):
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        # Structure initializers
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_bool,
            self.n_features,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        return toolbox

    def _default_toolbox(self):
        toolbox = self._init_toolbox()
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.evaluate)
        return toolbox

    def get_final_scores(self, pop, fits):
        self.final_fitness = list(zip(pop, fits))

    def generate(self, n_pop, cxpb=0.5, mutxpb=0.2, ngen=5, set_toolbox=False):

        """
        Generate evolved population
        Parameters
        -----------
            n_pop : {int}
                    population size
            cxpb  : {float}
                    crossover probablity
            mutxpb: {float}
                    mutation probablity
            n_gen : {int}
                    number of generations
            set_toolbox : {boolean}
                          If True then you have to create custom toolbox before calling
                          method. If False use default toolbox.
        Returns
        --------
            Fittest population
        """

        if self.verbose == 1:
            print(
                "Population: {}, crossover_probablity: {}, mutation_probablity: {}, total generations: {}".format(
                    n_pop, cxpb, mutxpb, ngen
                )
            )

        if not set_toolbox:
            self.toolbox = self._default_toolbox()
        else:
            raise Exception(
                "Please create a toolbox.Use create_toolbox to create and register_toolbox to register. Else set set_toolbox = False to use defualt toolbox"
            )
        pop = self.toolbox.population(n_pop)
        CXPB, MUTPB, NGEN = cxpb, mutxpb, ngen

        # Evaluate the entire population
        print("EVOLVING.......")
        fitnesses = list(map(self.toolbox.evaluate, pop))

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(NGEN):
            print("\n\n-- GENERATION {} --\n\n".format(g + 1))
            offspring = self.toolbox.select(pop, len(pop))
            self.fitness_in_generation[str(g + 1)] = max(
                [ind.fitness.values[0] for ind in pop]
            )
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            weak_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, weak_ind))
            for ind, fit in zip(weak_ind, fitnesses):
                ind.fitness.values = fit
            print("Evaluated %i individuals" % len(weak_ind))

            pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5
        if self.verbose == 1:
            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)

        print("-- Only the fittest survives --")

        self.best_ind = tools.selBest(pop, 1)[0]
        print(
            "Best individual is %s, %s" % (self.best_ind, self.best_ind.fitness.values)
        )
        self.get_final_scores(pop, fits)

        return pop


if __name__ == '__main__':
    files = ['mm', 'cancer', 'heart', 'cc', 'winet', 'irist', 'park', 'spambase', 'kdd99']
    file = 'mm'
    sortData = True
    header = False
    asc = True
    safeType = 0
    dangerType = 1
    iterNum = 1
    fname = ''

    X, y = test.import_data_ga(file=file + '.csv', sortData=sortData, asc=asc, header=header)

    model = DCA(weightMat=DCA.weights_paper_0, safeType=safeType, dangerType=dangerType, cellPoolNum=100,
                    antigenCellNum=10, iterNum=iterNum, fname=fname, solver='ga', predictType=True)

    t_start_time = time.time()

    ga = FeatureSelectionGA(model, X, y)
    pop = ga.generate(10)

    t_end_time = time.time()

    print('【用时：】', str((t_end_time - t_start_time)), 'sec')
