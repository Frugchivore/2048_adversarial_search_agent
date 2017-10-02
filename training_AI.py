# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 12:55:36 2017

@author: n.soungadoy
"""
import pickle
from datetime import datetime
from multiprocessing import freeze_support
from concurrent.futures import ProcessPoolExecutor
from concurrent import futures
import random

import logging
import numpy as np

from bayes_opt import BayesianOptimization

from PlayerAI_3 import Heuristic, PlayerAI, IDAlphaBetaSearch
from BaseDisplayer_3 import BaseDisplayer
from SilentGameManager_3 import GameManager
from ComputerAI_3 import ComputerAI


logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] - %(asctime)s - %(message)s')
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)


init_vals =  {'target':[2048],
                'moves_w': [3.9853620335592361],
                 'max_w': [2.2512627197409776],
                 'cells_w': [1.6866211628779304],
                 'mergers_w': [1.3878007134328434],
                 'smooth_w': [3.9512616409294972],
                 'mono_w': [2.9205369171896551],
                 'dense_w' : [0]}


search_space_old = {'max_w':(0,5),
                 'cells_w':(0,5),
                 'moves_w':(0,5),
                 'mergers_w':(0,5),
                 'smooth_w':(0,5),
                 'mono_w':(0,5),
                 'dense_w': (0,5)}

search_space = {'max_w':(0,5),
                 'cells_w':(0,5),
                 'smooth_w':(0,5),
                 'mono_w':(0,5)}

params = {'init_points':10, 'n_iter':30}

def training_task(individual, i=None):
    gameManager = GameManager()
    playerAI  	= PlayerAI(heuristic=individual)
    computerAI  = ComputerAI()
    displayer 	= BaseDisplayer() # Dummy. we don't want to have anything displayed.

    gameManager.setDisplayer(displayer)
    gameManager.setPlayerAI(playerAI)
    gameManager.setComputerAI(computerAI)

    fitness = gameManager.start()
    return fitness, individual, i

class BayesianOptimizer:

    def __init__(self,
                 template_cls=Heuristic,
                 max_workers=None,
                 cv_size=5,
                 search_space=search_space,
                 init_vals=init_vals,
                 params=params):
        self.template_cls = template_cls
        self.max_workers = max_workers
        self.cv_size = cv_size
        self.search_space = search_space
        self.init_vals = init_vals
        self.params = params
        self.bo = None

    def target(self, max_w, cells_w, smooth_w, mono_w):
        evalutions = []
        f = self.template_cls()
#        if isinstance(weights, np.ndarray):
#            weights = weights.tolist()

        weights = [max_w, cells_w, smooth_w, mono_w]

        f.weights = weights
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for i in range(self.cv_size):
                evalutions.append(
                            executor.submit(training_task, f, i)
                            )
            cv_scores = 0
            for completed in futures.as_completed(evalutions):
                fit, func, i = completed.result()
                cv_scores += fit

        return cv_scores  / self.cv_size

    def train(self):
        self.bo = BayesianOptimization(self.target, self.search_space)
#        self.bo.initialize(self.init_vals)
        self.bo.maximize(**self.params)
        logging.info(self.bo.res['max'])
        self.bo.points_to_csv("weights.csv")



    def save_state(self):

        with open("bayes_train_state.pkl", "wb") as f:
            pickler = pickle.Pickler(f)
            pickler.dump(self.bo)

    def load_state(self):
        with open("train_state.pkl", "rb") as f:
            unpickler = pickle.Unpickler(f)
            data = unpickler.load()
        self.bo = data


class GeneticAlgorithm:

    def __init__(self,
                 template_cls=Heuristic,
                 max_workers=None,
                 population_size=20,
                 epochs=12,
                 cross_validation=3,
                 p_mutation=0.1,
                 save_steps=2,
                 dropout=3
                 ):
        self.max_workers = max_workers
        self.epochs = epochs
        self.cross_validation=cross_validation
        self.population_size = population_size
        self.template_cls = template_cls
        self.p_mutation = p_mutation
        self.dropout = dropout
        self.save_steps = save_steps
        self.state_space = np.arange(0.1, 5, 0.1)
        self.population = self.init_population()
        self.fitness_func = np.zeros_like(self.population, dtype=np.float)
        self.proba = np.ones(self.population_size, dtype=np.float) / self.population_size


    def init_population(self):
        new_pop = []
        for i in range(self.population_size):
            individual = self.template_cls()
            weights = individual.weights
            new_weights = [np.random.choice(self.state_space) for w in weights]
            individual.weights = new_weights
            new_pop.append(individual)
        return new_pop

    def update_proba(self):
        total_fit = self.fitness_func.sum()
        self.proba = self.fitness_func / total_fit
        # We perform culling: the n worst individual are forbidden from reproducing
        min_idx = np.argsort(self.fitness_func)
        self.proba[min_idx[:self.dropout]] = 0
        self.proba /= self.proba.sum()

#        logging.debug("proba: {}".format(self.proba))

    def rand_select(self):
        return np.random.choice(self.population, p=self.proba)

    def mutate(self, child):
        # We assume the weights distributed as a gaussian centered around
        # the previous value.
        idx = random.randrange(len(child.weights))
        w = np.random.choice(self.state_space)
#        w = child.weights[idx]
        child.weights[idx] = w
        return child

    def crossover(self, x, y):
#        idx = random.randrange(len(x.weights))
        n_feat = len(x.weights)
        idx = np.random.choice(range(n_feat), n_feat//2, replace=False)
        child = self.template_cls()
#        child.weights = x.weights[:idx] + y.weights[idx:]

        child.weights = list(x.weights)
        for i in idx:
            child.weights[i] = y.weights[i]
        return child


    def evaluate_fit(self)  :
        fs = []
        fitness_func = np.zeros_like(self.population, dtype=np.float)
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for i, individual in enumerate(self.population):
                for j in range(self.cross_validation):
                    fs.append(
                            executor.submit(training_task, individual, i)
                            )
            logging.info("all tasks submitted")
            for completed in futures.as_completed(fs):
                fit, individual, i = completed.result()

                fitness_func[i] += fit / self.cross_validation
#                logging.debug("i: {}, fit: {}, fitness: {}".format(i, fit, fitness_func[i]))

        self.fitness_func = fitness_func

#        logging.debug(self.fitness_func)

        self.update_proba()
        maxfit = self.fitness_func.max()
        idxmax = np.argmax(self.fitness_func)

        logging.debug("idxmax: {}, maxfit: {}".format(idxmax, maxfit))
#        for k, individual in enumerate(self.population):
#            logging.debug("i: {}, w: {}".format(k, individual.weights))

        fittest = self.population[idxmax]
        logging.info("evaluation complete, maxfit: {}, maxweigths: {}".format(maxfit, fittest.weights))
        return maxfit, fittest

    def train(self):
        start_time = datetime.now()
        logging.info("starting training at {}".format(start_time))
        for step in range(self.epochs):
            logging.info("epoch: {}".format(step+1))
            logging.info("evaluating fitness...")
            maxfit, fittest = self.evaluate_fit()

            new_pop = [fittest]
            for i in range(self.population_size-1):
                x = self.rand_select()
                y = self.rand_select()
                child = self.crossover(x, y)
                p = random.random()

                if p > self.p_mutation:
                    child = self.mutate(child)
                new_pop.append(child)

            self.population = new_pop
            if (step+1) % self.save_steps == 0:
                logging.info("saving state")
                self.save_state()

        logging.info("final evaluation")
        logging.info("evaluating fitness...")
        maxfit, fittest = self.evaluate_fit()

        end_time = datetime.now()
        duration = end_time - start_time
        logging.info("training session completed successfully at {}".format(end_time))
        logging.info("total duration: {}".format(duration))
        logging.info("maxfit: {}, reached with weights: {}".format(maxfit, fittest.weights))
        self.save_weights(fittest.weights)
        return fittest

    def load_state(self):
        with open("train_state.pkl", "rb") as f:
            unpickler = pickle.Unpickler(f)
            data = unpickler.load()
        self.population = data['population']
        self.population_size = data['population_size']
        self.fitness_func = data['fitness_func']
        self.update_proba()

    def save_state(self):
        payload = {}
        payload['population'] = self.population
        payload['population_size'] = self.population_size
        payload['fitness_func'] = self.fitness_func

        with open("train_state.pkl", "wb") as f:
            pickler = pickle.Pickler(f)
            pickler.dump(payload)

    def save_weights(self, weigths):
        with open("weigths.csv", "w") as f:
            f.write(str(weigths))




def main():
#    ga_trainer = GeneticAlgorithm()
#    ga_trainer.load_state()
#    ga_trainer.train()
    bayes_trainer = BayesianOptimizer()
    bayes_trainer.train()
    bayes_trainer.save_state()
#    bayes_trainer.target(2.,  3.,  1.,  4.,  3.,  4.)

if __name__ == "__main__":
    freeze_support()
    main()

