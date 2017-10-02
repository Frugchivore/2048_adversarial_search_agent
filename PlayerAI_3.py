# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 23:02:39 2017

@author: n.soungadoy
"""

import random
import time
from itertools import product
from collections import Counter
from math import log2
from BaseAI_3 import BaseAI
weights = [ 1, 1.7, 0, 1, 1, 0.75]
weights3 = [ 3.3024,
                        1.0964952148933955,
                        1.6891185891270943,
                        1.0946,
                         0.3674,
                        2.26912403290832221,
                         1.7570]
weights2 = [2.3577414947122675,
                        3.0964952148933955,
                        1.6891185891270943,
                        4.3092492308377279,
                        2.0616074687186761,
                        3.26912403290832221]
scores = [128, 256, 64, 512, 256, 512]

scores = {'max_params':
                {'max_w': 2.3577414947122675,
                 'cells_w': 3.0964952148933955,
                 'mergers_w': 1.3092492308377279,
                 'moves_w': 1.6891185891270943,
                 'smooth_w': 1.0616074687186761,
                 'mono_w': 0.26912403290832221},
        'max_val': 490.66666666666669}

#scores {'max_val': 665.60000000000002,
#        'max_params': {'max_w':  3.7331081489086548,
#    'mono_w': 0.80988229148170943,
#    'smooth_w': 1.3861795719789898,
#    'cells_w': 3.2927997573902923}
#    }
class Heuristic:
    def __init__(self):
        self.weights = [1.0, 3, 0.25, 2.75, 0.75, 3.5]

    def __call__(self, grid):

        f1 = log2(grid.getMaxTile())
        f2 = len(grid.getAvailableCells())
        f3 = len(grid.getAvailableMoves())

        f4, f5 = self.mergers_and_smoothness(grid)
#        f4 = 0
#        f5 = self.smoothness(grid)
        f6 = self.monotonicity(grid)
#        f7 = 0 #self.density(grid)
        features = [f1, f2, f3, f4, f5, f6]

        h = 0
        for w, f in zip(self.weights, features):
            h += w * f

        return h

    def counter(self, grid):
        c = Counter

        for x in range(grid.size):
            for y in range(grid.size):
                c[grid.map[x][y]] += 1
        return sum(v for v in c.values() if v > 1)

    def density(self, grid):
        i = 0
        average = 0
        for x in range(grid.size):
            for y in range(grid.size):
                i += 1
                mass = log2(grid.map[x][y]) if grid.map[x][y] > 0 else 0
                average += mass * i
        return average

    def monotonicity(self, grid):
        # [rightward, leftward]
        total_rows = [[0,0],[0,0],[0,0],[0,0]]
        total_cols = [[0,0],[0,0],[0,0],[0,0]]
        # rows
        for x in range(grid.size):
            for y in range(grid.size-1):
                curr_value = grid.map[x][y]
                curr_value = log2(curr_value) if curr_value > 0 else 0
                next_value = grid.map[x][y+1]
                next_value = log2(next_value) if next_value > 0 else 0

#                if curr_value > next_value:
#                    total[0] += next_value - curr_value
#                elif curr_value < next_value:
#                    total[1] += curr_value - next_value

                if curr_value > next_value:
                    total_rows[x][0] += next_value - curr_value
                elif curr_value < next_value:
                    total_rows[x][1] += curr_value - next_value


        # columns
        for y in range(grid.size):
            for x in range(grid.size-1):
                curr_value = grid.map[x][y]
                curr_value = log2(curr_value) if curr_value > 0 else 0
                next_value = grid.map[x+1][y]
                next_value = log2(next_value) if next_value > 0 else 0

#                if curr_value > next_value:
#                    total[2] += next_value - curr_value
#                elif curr_value < next_value:
#                    total[3] += curr_value - next_value

                if curr_value > next_value:
#                    mono_idx += 1
                    total_cols[y][0] += next_value - curr_value
                elif curr_value < next_value:
                    total_cols[y][1] += curr_value - next_value

        score = 0
        for row, col in zip(total_rows, total_cols):
             score += max(row[0], row[1])
             score += max(col[0], col[1])
        return score
#        return max(total[0] , total[1]) + max(total[2] , total[3])


    def monotonicity2(self, grid):
        # [rightward, leftward]
        total_rows = [0,0,0,0]
        total_cols = [0,0,0,0]
        # rows
        for x in range(grid.size):
            previous = 0
            for y in range(grid.size-1):
                curr_value = grid.map[x][y]
                curr_value = log2(curr_value) if curr_value > 0 else 0
                next_value = grid.map[x][y+1]
                next_value = log2(next_value) if next_value > 0 else 0

                diff = next_value - curr_value
                if diff * previous >= 0:
                    total_rows[x] += abs(diff)
                else:
                    total_rows[x] = 0
                    break


        # columns
        for y in range(grid.size):
            previous = 0
            for x in range(grid.size-1):
                curr_value = grid.map[x][y]
                curr_value = log2(curr_value) if curr_value > 0 else 0
                next_value = grid.map[x+1][y]
                next_value = log2(next_value) if next_value > 0 else 0


                diff = next_value - curr_value
                if diff * previous >= 0:
                    total_cols[y] += abs(diff)
                else:
                    total_cols[y] = 0
                    break


#        print(total_rows)
#        print(total_cols)
        return max(total_rows) + max(total_cols)

#        return max(total[0] , total[1]) + max(total[2] , total[3])


    def mergers(self, grid):
        merge_count = 0
        for x in range(grid.size):
            for y in range(grid.size):
                Y = y+1
                if Y < grid.size:
                    diffLR = abs(grid.map[x][y] - grid.map[x][Y])
                    if diffLR == 0: #Merge opportunity
                        merge_count += 1

                X = x+1
                if X < grid.size:
                    diffUD = abs(grid.map[x][y] - grid.map[X][y])
                    if diffUD == 0: #Merge opportunity
                        merge_count += 1

        return merge_count

    def farthest_value(self, grid, pos, vector):
        x, y = pos
        i, j = vector
        target_value = 0
        while x < grid.size and y < grid.size and grid.map[x][y] != 0:
            target_value = grid.map[x][y]
            x+=i
            y+=j

        return target_value

    def smoothness(self, grid):
        smoothness = 0
        for x in range(grid.size):
            for y in range(grid.size):
                if grid.map[x][y] != 0:
                    value = log2(grid.map[x][y])
                    for vector in [(1,0), (0,1)]:
                        target_value = self.farthest_value(grid, (x, y), vector)

                        if target_value != 0:
                            target_value = log2(target_value)
                            smoothness -= abs(value - target_value)

        return smoothness


    def mergers_and_smoothness(self, grid):
        merge_count = 0
        smoothness = 0
        for x in range(grid.size):
            for y in range(grid.size):
                if grid.map[x][y] != 0:
                    value = log2(grid.map[x][y])
                    for vector in [(1,0), (0,1)]:
                        target_value = self.farthest_value(grid, (x, y), vector)

                        if target_value != 0:
                            target_value = log2(target_value)
                            smoothness -= abs(value - target_value)

                Y = y+1
                if Y < grid.size:
                    diffLR = abs(grid.map[x][y] - grid.map[x][Y])
                    if diffLR == 0: #Merge opportunity
                        merge_count += 1

                X = x+1
                if X < grid.size:
                    diffUD = abs(grid.map[x][y] - grid.map[X][y])
                    if diffUD == 0: #Merge opportunity
                        merge_count += 1

        return merge_count, smoothness

class IDMinimaxSearch:

    def __init__(self, heuristic, time_limit):
        self.heuristic = heuristic
        self.time_limit = time_limit
        self._stopped = None
        self.reset()

    @property
    def stopped(self):
        return self._stopped

    @stopped.setter
    def stopped(self, value):
        self._stopped = value

    def reset(self):
        self.stopped = False
        self.depth = 0
        self.maxdepth = 1

    def __call__(self, grid):
        return self.minimax(grid)

    def elapsed(self):
        return time.clock() - self.start

    def cutoff_test(self, grid):
        flag = False
        if self.elapsed() >= self.time_limit:
            self.stopped = True
            flag = True
        if self.depth >= self.maxdepth:
            flag = True
        moves = grid.getAvailableMoves()
        if not moves:
            flag = True
        return flag


    def minimax(self, grid):
        self.start = time.clock()
        self.reset()
        v = (float('-inf'), None)
        while not self.stopped:
            self.depth = 0

            for move in grid.getAvailableMoves():
                g = grid.clone()
                g.move(move)
                v = max(v , (self._minimax(g, False), move))
            self.maxdepth += 1
        return v[1]

    def _minimax(self, grid, MAX):
        if self.cutoff_test(grid):
            return self.heuristic(grid)

        if MAX:
            self.depth += 1
            v =  float('-inf')
            for move in grid.getAvailableMoves():
                g = grid.clone()
                g.move(move)
                v = max(v, self._minimax(g, False))
                if self.stopped:
                    break
        else:
            v =  float('inf')
            cells = grid.getAvailableCells()
            iterator = product(cells, [2, 4])
            for cell, tile_value in iterator:
                g = grid.clone()
                g.setCellValue(cell, tile_value)
                v = min(v, self._minimax(g, True))
                if self.stopped:
                    break
        return v


class IDAlphaBetaSearch:

    def __init__(self, time_limit, heuristic):
        self.heuristic = heuristic if heuristic else Heuristic()
        self.grid_cache = {}
        self.eval_cache = {}
        self._stopped = None
        self.time_limit = time_limit
        self.reset()

    @property
    def stopped(self):
        return self._stopped

    @stopped.setter
    def stopped(self, value):
        self._stopped = value

    def reset(self):
        self.grid_cache = {}
        self.eval_cache = {}

        self.stopped = False
        self.depth = 0
        self.maxdepth = 3

    def __call__(self, grid):
        return self.alpha_beta_pruning(grid)

    def elapsed(self):
        return time.clock() - self.start

    def cutoff_test(self, grid):
        flag = False
        if self.elapsed() >= self.time_limit:
            self.stopped = True
            flag = True
        if self.depth >= self.maxdepth:
            flag = True
        moves = grid.getAvailableMoves()
        if not moves:
            flag = True

        return flag


    def alpha_beta_pruning(self, grid):
        self.start = time.clock()
        self.reset()
        best = (float('-inf'), None)
        while not self.stopped:
            self.depth = 0
            alpha = float('-inf')
            beta = float('inf')

            best = max(best, self._alpha_beta_pruning(grid, alpha, beta))

            self.maxdepth += 1
        return best[1]


    def _alpha_beta_pruning(self, grid, alpha, beta):
        self.depth += 1
        best = (float('-inf'), None)

        if not self.grid_cache.get(grid):
            self.grid_cache[grid] = []
            for move in grid.getAvailableMoves():
                g = grid.clone()
                g.move(move)
                self.grid_cache[grid].append((move, g))

        for move, g in self.grid_cache[grid]:
            g = grid.clone()
            g.move(move)
            best = max(best, (self.min_value(g, alpha, beta), move))
            if best[0] >= beta:
                return best
            alpha = max(best[0], alpha)
            if self.stopped:
                break

        return best

    def max_value(self, grid, alpha, beta):
        if self.cutoff_test(grid):
            if not self.eval_cache.get(grid):
                self.eval_cache[grid] = self.heuristic(grid)
            return self.eval_cache[grid]



        self.depth += 1
        v = float('-inf')

        if not self.grid_cache.get(grid):
            self.grid_cache[grid] = []
            for move in grid.getAvailableMoves():
                g = grid.clone()
                g.move(move)
                self.grid_cache[grid].append((move, g))

        for move, g in self.grid_cache[grid]:
            g = grid.clone()
            g.move(move)
            v = max(v, self.min_value2(g, alpha, beta))

            alpha = max(v, alpha)
            if beta <= alpha:
                break

            if self.stopped:
                break

        return v

    def min_value(self, grid, alpha, beta):
        if self.cutoff_test(grid):
            if not self.eval_cache.get(grid):
                self.eval_cache[grid] = self.heuristic(grid)
            return self.eval_cache[grid]


        self.depth += 1
        v = float('inf')
        if not self.grid_cache.get(grid):
            self.grid_cache[grid] = []
            cells = grid.getAvailableCells()
            iterator = product(cells, [2, 4])
            for cell, tile_value in iterator:
                g = grid.clone()
                g.setCellValue(cell, tile_value)
                self.grid_cache[grid].append(g)

        for g in self.grid_cache[grid]:

            v = min(v, self.max_value(g, alpha, beta))

            beta = min(v, beta)
            if beta <= alpha:
                break

            if self.stopped:
                break

        return v

    def min_value2(self, grid, alpha, beta):
        if self.cutoff_test(grid):
            if not self.eval_cache.get(grid):
                self.eval_cache[grid] = self.heuristic(grid)
            return self.eval_cache[grid]


#        self.depth += 1
        v = float('inf')
        if not self.grid_cache.get(grid):
            self.grid_cache[grid] = []
            cells = grid.getAvailableCells()
            iterator = product(cells, [2, 4])
            for cell, tile_value in iterator:
                g = grid.clone()
                g.setCellValue(cell, tile_value)
                self.grid_cache[grid].append(g)

        worst = float('inf')
        worst_g = None
        for g in self.grid_cache[grid]:
            if not self.eval_cache.get(g):
                self.eval_cache[g] = self.heuristic(g)
            if worst > self.eval_cache[g]:
                worst = self.eval_cache[g]
                worst_g = g
#            worst = min(worst, (self.eval_cache[g], g))

        v = min(v, self.max_value(worst_g, alpha, beta))

        beta = min(v, beta)
#        if beta <= alpha:
#            break
#
#        if self.stopped:
#            break

        return v




    def alpha_beta_pruning2(self, grid):
        self.start = time.clock()
        self.reset()
        v = (float('-inf'), None)
        while not self.stopped:
            self.depth = 0
            alpha = float('-inf')
            beta = float('inf')
            if not self.grid_cache.get(grid):
                self.grid_cache[grid] = []
                for move in grid.getAvailableMoves():
                    g = grid.clone()
                    g.move(move)
                    self.grid_cache[grid].append((move, g))

#            for move in grid.getAvailableMoves():
            for move, g in self.grid_cache[grid]:
#                g = grid.clone()
#                g.move(move)

                v = max(v ,
                        (self._alpha_beta_pruning2(g, False, alpha, beta), move)
                )

            self.maxdepth += 1
        return v[1]

    def _alpha_beta_pruning2(self, grid, MAX, alpha, beta):
        if self.cutoff_test(grid):
            if not self.eval_cache.get(grid):
                self.eval_cache[grid] = self.heuristic(grid)

            return self.eval_cache[grid]
#            return  self.heuristic(grid)


        if MAX:
            self.depth += 1
            v =  float('-inf')
            if not self.grid_cache.get(grid):
                self.grid_cache[grid] = []
                for move in grid.getAvailableMoves():
                    g = grid.clone()
                    g.move(move)
                    self.grid_cache[grid].append((move, g))
#            for move in grid.getAvailableMoves():
#                g = grid.clone()
#                g.move(move)
            for move, g in self.grid_cache[grid]:
                v = max(v, self._alpha_beta_pruning2(g, False, alpha, beta))
                alpha = max(v, alpha)
                if  beta <= alpha:
                    return v

                if self.stopped:
                    break
        else:
            v =  float('inf')
            if not self.grid_cache.get(grid):
                self.grid_cache[grid] = []
                cells = grid.getAvailableCells()
                iterator = product(cells, [2, 4])
                for cell, tile_value in iterator:
                    g = grid.clone()
                    g.setCellValue(cell, tile_value)
                    self.grid_cache[grid].append(g)

#            cells = grid.getAvailableCells()
#            iterator = product(cells, [2, 4])
#            for cell, tile_value in iterator:
#                g = grid.clone()
#                g.setCellValue(cell, tile_value)
            for g in self.grid_cache[grid]:
                v = min(v, self._alpha_beta_pruning2(g, True, alpha, beta))
                beta = min(v, beta)
                if beta <= alpha:
                    return v

                if self.stopped:
                    break
        return v

class PlayerAI(BaseAI):

    def __init__(self, time_limit=0.099, heuristic=None):
        self.search = IDAlphaBetaSearch(time_limit, heuristic=None)

    def getMove(self, grid):
        return self.search(grid)

if __name__ == "__main__":
    from GameManager_3 import main
    main()
    print(Heuristic().weights)