## Project: 2048 Solver
____

This implementation of a 2048 solving agent was written as part of the Artificial Intelligence class from edX AI MicroMaster.
It uses various adversarial search methods to obtain the highest score possible in the game. A Heuristic function can be trained
either via a Genetic Algorithm solver I wrote or using the Bayesian Optimization library bayes_opt.

#### Structure of the project:

It can be assumed that all the Python files not listed below were provided by edX.

- [*PlayerAI_3.py*](./PlayerAI_3.py): Contains the Agent, the various Solvers available as well as the Heuristic class.
- [*SilentGameManager_3.py*](./SilentGameManager_3.py): I modified the default GameManager to not display the game board and score on screen. This speeds up the execution and allows us to
train the AI.
- [*training_AI.py*](./training_AI.py): Contains the classes BayesianOptimizer, GeneticAlgorithm and code to start a training task. The goal being to find the appropriate weights in our heuristic function.
