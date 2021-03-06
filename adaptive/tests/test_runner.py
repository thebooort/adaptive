# -*- coding: utf-8 -*-

import asyncio

from ..learner import Learner2D
from ..runner import simple, BlockingRunner, AsyncRunner, SequentialExecutor


def test_nonconforming_output():
    """Test that using a runner works with a 2D learner, even when the
    learned function outputs a 1-vector. This tests against the regression
    flagged in https://gitlab.kwant-project.org/qt/adaptive/issues/58.
    """

    def f(x):
        return [0]

    def goal(l):
        return l.npoints > 10

    learner = Learner2D(f, [(-1, 1), (-1, 1)])
    simple(learner, goal)

    learner = Learner2D(f, [(-1, 1), (-1, 1)])
    BlockingRunner(learner, goal, executor=SequentialExecutor())

    learner = Learner2D(f, [(-1, 1), (-1, 1)])
    runner = AsyncRunner(learner, goal, executor=SequentialExecutor())
    asyncio.get_event_loop().run_until_complete(runner.task)
