import os
import time
import numpy as np

from thirdparty.revde.algorithms.recombination import DifferentialRecombination
from thirdparty.revde.algorithms.selections import SelectBest

class DifferentialEvolution():
    def __init__(self, x0, num_env, de_type='de', bounds=(0,1), params=None, output_dir="./results"):
        self.population_size = x0.shape[0]
        self.num_env = num_env
        self.gen = 0
        self.x_new = x0
        self.x = x0[0:0]

        self.f = x0[0:0]

        self.f_best_so_far = []
        self.x_best_so_far = []

        self.differential = DifferentialRecombination(type=de_type, bounds=bounds, params=params)
        self.selection = SelectBest()

        self.directory_name = output_dir
        self.fitnesses = []
        self.genomes = []

    def add_eval(self, fitness):
        current_genome, self.x_new = self.x_new[:1], self.x_new[1:]
        self.x = np.vstack((self.x, current_genome))
        self.f = np.append(self.f, fitness)
        if self.x_new.shape[0] == 0:
            return True
        return  False

    def get_new_genomes(self):
        if self.x_new.shape[0] == 0:
            self.new_pop()
            if type(self.x_new) == tuple:
                self.x_new = np.concatenate(self.x_new, 0)
        return self.x_new[:self.num_env]

    def new_pop(self):
        x, f = self.selection.select(self.x, self.f, population_size=self.population_size)

        f_min = np.min(f)
        if self.f_best_so_far == [] or f_min < self.f_best_so_far[-1]:
            self.f_best_so_far.append(f_min)

            ind_min = np.argmin(f)

            self.x_best_so_far.append(x[[ind_min]])
        else:
            self.x_best_so_far.append(self.x_best_so_far[-1])
            self.f_best_so_far.append(self.f_best_so_far[-1])

        self.x_new, _ = self.differential.recombination(x)
        self.x = x
        self.f = f

        self.genomes.append(self.x)
        self.fitnesses.append(self.f)

        self.gen += 1
        # print(f"New population, gen: {self.gen} \t | \t {time.clock()}")
    def save_results(self):
        np.save(self.directory_name + '/' + 'fitnesses', np.array(self.fitnesses))
        np.save(self.directory_name + '/' + 'genomes', np.array(self.genomes))

        np.save(self.directory_name + '/' + 'f_best', np.array(self.f_best_so_far))
        np.save(self.directory_name + '/' + 'x_best', np.array(self.x_best_so_far))


    def save_checkpoint(self):
        self.save_results()
        np.save(self.directory_name + '/' + 'last_x_new', np.array(self.x_new))
        np.save(self.directory_name + '/' + 'last_x', np.array(self.x))
        np.save(self.directory_name + '/' + 'last_f', np.array(self.f))

#%%
from deap import base
from deap import creator
from deap import tools
import pickle
from matplotlib import pyplot as plt
import copy

class NSGA2():
    def __init__(self, x0, num_env, de_type='de', bounds=(0,1), params=None, output_dir="./results"):
        self.population_size = x0.shape[0]
        self.num_env = num_env
        self.gen = 0
        self.x_new = x0
        self.x = x0[0:0]
        self.f = x0[0:0,:3]

        self.f_best_so_far = []
        self.x_best_so_far = []

        self.directory_name = output_dir
        self.fitnesses = []
        self.genomes = []

        creator.create("FitnessesMax", base.Fitness, weights=(1.0,) * 3)
        creator.create("Individual", list, fitness=creator.FitnessesMax)

        self.toolbox = base.Toolbox()
        # self.logbook = None
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "gait", "rotr", "rotl"
        self.logbook.chapters["gait"].header = "avg", "std", "max"
        self.logbook.chapters["rotr"].header = "avg", "std", "max"
        self.logbook.chapters["rotl"].header = "avg", "std", "max"

        self.toolbox.register("genotype", self.rand_ind)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.genotype, 1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual, n=self.population_size)
        self.pop = self.toolbox.population()
        self.x_new = np.array(self.pop).squeeze()
        self.invalid_ind = copy.deepcopy(self.pop)

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("select", tools.selNSGA2, k=int(self.population_size/2))
        self.toolbox.register("mutate", tools.mutGaussian, indpb=0.05, mu=params['MUT_MU'], sigma=params['MUT_SIGMA'])
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.CXPB = params['CXPB']
        self.MUTPB = params['MUTPB']

        stats_gait = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_rotr = tools.Statistics(lambda ind: ind.fitness.values[1])
        stats_rotl = tools.Statistics(lambda ind: ind.fitness.values[2])
        self.stats = tools.MultiStatistics(gait=stats_gait, rotr=stats_rotr, rotl=stats_rotl)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("max", np.max, axis=0)

    def rand_ind(self):
        return np.random.uniform(-1, 1, (self.x.shape[1]))

    def evaluate(self, individual):
        individual = individual[0]
        index = np.where((self.x == individual).all(axis=1))
        return self.f[index][-1]

    def add_eval(self, fitness):
        current_genome, self.x_new = self.x_new[:1], self.x_new[1:]
        self.x = np.vstack((self.x, current_genome))
        self.f = np.vstack((self.f, fitness))

    def get_new_genomes(self):
        if self.x_new.shape[0] == 0:
            self.new_pop()
            if type(self.x_new) == tuple:
                self.x_new = np.concatenate(self.x_new, 0)
        return self.x_new[:self.num_env]

    def new_pop(self):
        fitnesses = list(map(self.toolbox.evaluate, self.pop))
        for ind, fit in zip(self.invalid_ind, fitnesses):
            ind.fitness.values = fit
        self.pop[:] = copy.deepcopy(self.invalid_ind)
        # best = self.f[np.argmin(self.toolbox.evaluate(x) for x in self.pop)]
        ind = np.argmax(self.f, axis=0)
        # print(round(self.f[ind[0],0], 4),
        #       round(self.f[ind[1],1], 4),
        #       round(self.f[ind[2],2], 4))

        offspring = self.toolbox.select(self.pop)

        # Clone the selected individuals
        mutants = list(map(self.toolbox.clone, offspring))

        # # Apply crossover and mutation on the offspring
        # for child1, child2 in zip(offspring[::2], offspring[1::2]):
        #     if np.random.random() < self.CXPB:
        #         self.toolbox.mate(child1[0], child2[0])
        #         del child1.fitness.values
        #         del child2.fitness.values

        for mutant in mutants:
            if np.random.random() <= self.MUTPB:
                self.toolbox.mutate(mutant[0])
                # del mutant.fitness.values

        # offspring = self.toolbox.select(offspring+mutants)

        # Evaluate the individuals with an invalid fitness
        self.invalid_ind = offspring + mutants
        self.x_new = np.array([ind[0] for ind in self.invalid_ind ])
        self.gen += 1

    def log_results(self, generation, new=True):
        record = self.stats.compile(self.pop)
        self.logbook.record(gen=generation, **record)
        print(self.logbook.stream)
        with open(self.directory_name + '/evolution_summary.pkl', 'wb') as output:
            pickle.dump(self.logbook, output, pickle.HIGHEST_PROTOCOL)
        self.plots_summary()

    def read_logbook(self):
        with open(self.directory_name+'/evolution_summary.pkl', 'rb') as input:
            self.logbook = pickle.load(input)
            print(self.logbook)

    def plots_summary(self):
        gen = self.logbook.select("gen")
        gait_avg = self.logbook.chapters["gait"].select("avg")
        gait_std = self.logbook.chapters["gait"].select("std")
        rotr_avg = self.logbook.chapters["rotr"].select("avg")
        rotr_std = self.logbook.chapters["rotr"].select("std")
        rotl_avg = self.logbook.chapters["rotl"].select("avg")
        rotl_std = self.logbook.chapters["rotl"].select("std")

        fig, ax1 = plt.subplots()
        ax1.plot([int(x) for x in gen], gait_avg, "k-", label="Gait")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("m/s", color="k")
        # ax1.set_ylim(ymin=0, ymax=6)
        for tl in ax1.get_yticklabels():
            tl.set_color("k")

        ax1.fill_between(gen, np.array(gait_avg) - np.array(gait_std),
                         np.array(gait_avg) + np.array(gait_std), alpha=0.2, facecolor='#000000')

        ax2 = ax1.twinx()
        ax2.plot(gen, rotr_avg, "b-", label="Rot-r")
        ax2.set_ylabel("rad/s", color="b")
        # ax2.set_ylim(ymin=0, ymax=6.28)
        for tl in ax2.get_yticklabels():
            tl.set_color("b")

        ax2.fill_between(gen, np.array(rotr_avg) - np.array(rotr_std),
                         np.array(rotr_avg) + np.array(rotr_std), alpha=0.2, facecolor='#FF9999')

        ax3 = ax1.twinx()
        ax3.plot(gen, rotl_avg, "r-", label="Rot-l")
        ax3.set_ylabel("rad/s", color="r")
        # ax3.set_ylim(ymin=0, ymax=6.28)
        for tl in ax3.get_yticklabels():
            tl.set_color("r")

        ax3.fill_between(gen, np.array(rotl_avg) - np.array(rotl_std),
                         np.array(rotl_avg) + np.array(rotl_std), alpha=0.2, facecolor='#66B2FF')

        plt.savefig(self.directory_name + '/evolution_summary.png')
        plt.close()
        fig, ax  = plt.subplots()
        k_gen = self.gen//10
        gens = np.linspace(0, self.gen, k_gen)
        for i in range(k_gen):
            ind = int(gens[i]*self.population_size)
            ax.plot(self.f[ind:ind+self.population_size,1],
                    self.f[ind:ind+self.population_size,0],'o', label=f'gen_{gens[i]}', color=(1-i/k_gen,.1,.1))
        plt.savefig(self.directory_name + '/evolution_pareto.png')
        plt.close()