# -*- coding: utf-8 -*-
import random
import numpy as np
import pytest

from .test_learners import ring_of_fire, sphere_of_fire, generate_random_parametrization
from ..learner import LearnerND
from ..runner import replay_log, simple

def test_faiure_case_LearnerND():
    log = [
        ('ask', 4),
        ('tell', (-1, -1, -1), 1.607873907219222e-101),
        ('tell', (-1, -1, 1), 1.607873907219222e-101),
        ('ask', 2),
        ('tell', (-1, 1, -1), 1.607873907219222e-101),
        ('tell', (-1, 1, 1), 1.607873907219222e-101),
        ('ask', 2),
        ('tell', (1, -1, 1), 2.0),
        ('tell', (1, -1, -1), 2.0),
        ('ask', 2),
        ('tell', (0.0, 0.0, 0.0), 4.288304431237686e-06),
        ('tell', (1, 1, -1), 2.0)
    ]
    learner = LearnerND(lambda *x: x, bounds=[(-1, 1), (-1, 1), (-1, 1)])
    replay_log(learner, log)


# This sometimes fails and sometimes succeeds, my guess would be that this could 
# be due to a numerical precision error: 
# In the very beginning the loss of every interval is the same (as the function 
# is highly symetric), then by machine precision there will be some error and 
# then the simplex that has by accident some error that reduces the loss, 
# will be chosen.
@pytest.mark.xfail 
def test_learner_performance_is_invariant_under_scaling():
    kwargs = dict(bounds=[(-1, 1)]*2)
    f = generate_random_parametrization(ring_of_fire)

    control = LearnerND(f, **kwargs)

    xscale = 1000 * random.random()
    yscale = 1000 * random.random()

    l_kwargs = dict(kwargs)
    l_kwargs['bounds'] = xscale * np.array(l_kwargs['bounds'])
    learner = LearnerND(lambda x: yscale * f(np.array(x) / xscale), **l_kwargs)

    control._recompute_loss_whenever_scale_changes_more_than = 1
    learner._recompute_loss_whenever_scale_changes_more_than = 1

    npoints = random.randrange(1000, 2000)

    for n in range(npoints):
        cxs, _ = control.ask(1)
        xs, _ = learner.ask(1)

        control.tell_many(cxs, [control.function(x) for x in cxs])
        learner.tell_many(xs , [learner.function(x) for x in xs])
        if n > 100:
            assert np.isclose(learner.loss(), control.loss())
