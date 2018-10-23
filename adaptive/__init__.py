# -*- coding: utf-8 -*-
from contextlib import suppress

from .notebook_integration import (notebook_extension, live_plot,
                                   active_plotting_tasks)

from . import learner
from . import runner
from . import utils

from .learner import (BaseLearner, Learner1D, Learner2D, LearnerND,
					  AverageLearner, BalancingLearner, make_datasaver,
					  DataSaver, IntegratorLearner, AverageLearner2D)

with suppress(ImportError):
    # Only available if 'scikit-optimize' is installed
    from .learner import SKOptLearner

from .runner import Runner, AsyncRunner, BlockingRunner

from ._version import __version__
del _version

del notebook_integration  # to avoid confusion with `notebook_extension`
