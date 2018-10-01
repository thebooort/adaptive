# -*- coding: utf-8 -*-

from copy import deepcopy
from math import sqrt
import operator
import sys

import numpy as np

from ..notebook_integration import ensure_holoviews
from .learner1D import Learner1D


class AverageLearner1D(Learner1D):
    def __init__(self, function, bounds, loss_per_interval=None, weight=1):
        super().__init__(function, bounds, loss_per_interval)
        self._data = dict()  # {point: {seed: value}} mapping
        self.pending_points = dict()  # {point: {seed}}

        self.weight = weight
        self.min_values_per_point = 3

    @property
    def data(self):
        return {k: sum(v.values()) / len(v) for k, v in self._data.items()}

    @property
    def data_sem(self):
        return {k: self.standard_error(v.values()) for k, v in self._data.items()}

    def standard_error(self, lst):
        n = len(lst)
        if n < self.min_values_per_point:
            return sys.float_info.max
        sum_f_sq = sum(x**2 for x in lst)
        mean = sum(x for x in lst) / n
        numerator = sum_f_sq - n * mean**2
        if numerator < 0:
            # This means that the numerator is ~ -1e-15
            return 0
        std = sqrt(numerator / (n - 1))
        return std / sqrt(n)

    def mean_values_per_point(self):
        return np.mean([len(x.values()) for x in self._data.values()])

    def ask(self, n, tell_pending=True):
        """Return n points that are expected to maximally reduce the loss."""
        points, loss_improvements = self._ask_points_without_adding(n)
        points = [(p, 0) for p in points]

        y_scale = self._scale[1] or 1
        for (p, sem) in self.data_sem.items():
            if sem > 0:
                points.append((p, self.get_seed(p)))
                N = len(self._data[p])
                loss_improvement = (1 - sqrt(N - 1) / sqrt(N)) * sem
                loss_improvements.append(self.weight * loss_improvement / y_scale)

        points, loss_improvements = zip(*sorted(zip(points, loss_improvements),
            key=operator.itemgetter(1), reverse=True))

        points = list(points)[:n]
        loss_improvements = list(loss_improvements)[:n]

        if tell_pending:
            for p in points:
                self.tell_pending(p)

        return points, loss_improvements

    def tell(self, x_seed, y):
        x, seed = x_seed

        in_data = x in self._data

        # either it is a float/int, if not, try casting to a np.array
        if not isinstance(y, (float, int)):
            y = np.asarray(y, dtype=float)

        # Add point to the real data dict
        if x not in self._data:
            self._data[x] = {}
        self._data[x][seed] = y

        # remove from set of pending points
        if x in self.pending_points:
            self.pending_points[x].discard(seed)
            if not self.pending_points[x]:
                # pending_points[x] is now empty so delete the set()
                del self.pending_points[x]

        self._update_data_structures(x, y)

    def tell_pending(self, x_seed):
        x, seed = x_seed

        if x in self.pending_points:
            assert seed not in self.pending_points[x]

        # Add new point to 'pending_points'.
        if x not in self.pending_points:
            self.pending_points[x] = set()
        self.pending_points[x].add(seed)

        if x not in self.neighbors_combined:
            # If 'x' already exists then there is not need to update.
            self.update_neighbors(x, self.neighbors_combined)
            self.update_losses(x, real=False)

    def tell_many(self, xs, ys):
        # `super().tell_many(xs, ys)` will not work.
        for x, y in zip(xs, ys):
            self.tell(x, y)

    def get_seed(self, point):
        _data = self._data.get(point, {})
        pending_seeds = self.pending_points.get(point, set())
        seed = len(_data) + len(pending_seeds)
        if seed in _data or seed in pending_seeds:
            # means that the seed already exists, for example
            # when '_data[point].keys() | pending_points[point] == {0, 2}'.
            return (set(range(seed)) - pending_seeds - _data.keys()).pop()
        return seed

    def remove_unfinished(self):
        self.pending_points = {}
        self.losses_combined = deepcopy(self.losses)
        self.neighbors_combined = deepcopy(self.neighbors)

    def plot(self):
        hv = ensure_holoviews()
        scatter = super().plot()
        if self._data:
            xs, ys = zip(*sorted(self.data.items()))
            sem = self.data_sem
            spread = hv.Spread((xs, ys, [sem[x] for x in xs]))
            return scatter * spread
        else:
            return scatter
