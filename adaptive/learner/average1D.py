# -*- coding: utf-8 -*-

from copy import deepcopy
import sys

import numpy as np

from ..notebook_integration import ensure_holoviews
from .learner1D import Learner1D
from .average_mixin import AverageMixin


class AverageLearner1D(Learner1D):
    def __init__(self, function, bounds, loss_per_interval=None, weight=1):
        super().__init__(function, bounds, loss_per_interval)
        self._data = dict()  # {point: {seed: value}} mapping
        self.pending_points = dict()  # {point: {seed}}

        self.weight = weight
        self.min_values_per_point = 3

    def value_scale(self):
        return self._scale[1] or 1

    def _ask_points_without_adding(self, n):
        points, loss_improvements = super()._ask_points_without_adding(n)
        points = [(p, 0) for p in points]
        return points, loss_improvements

    def unpack_point(self, x_seed):
        return x, seed

    def tell(self, x_seed, y):
        x, seed = self.unpack_point(x_seed)

        in_data = x in self._data

        # either it is a float/int, if not, try casting to a np.array
        if not isinstance(y, (float, int)):
            y = np.asarray(y, dtype=float)

        self._add_to_data(x_seed)
        self._remove_from_to_pending(x_seed)
        self._update_data_structures(x, y)

    def tell_pending(self, x_seed):
        x, seed = self.unpack_point(x_seed)

        self._add_to_pending(x_seed)

        if x not in self.neighbors_combined:
            # If 'x' already exists then there is not need to update.
            self.update_neighbors(x, self.neighbors_combined)
            self.update_losses(x, real=False)

    def tell_many(self, xs, ys):
        # `super().tell_many(xs, ys)` will not work.
        for x, y in zip(xs, ys):
            self.tell(x, y)

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

    data = AverageMixin.data
    data_sem = AverageMixin.data_sem
    standard_error = AverageMixin.standard_error
    mean_values_per_point = AverageMixin.mean_values_per_point
    get_seed = AverageMixin.get_seed
    loss_per_existing_point = AverageMixin.loss_per_existing_point
    _add_to_pending = AverageMixin._add_to_pending
    _remove_from_to_pending = AverageMixin._remove_from_to_pending
    _add_to_data = AverageMixin._add_to_data
    ask = AverageMixin.ask
