from functools import partial

from .sinkgd_simple import SinkGD as SinkGD_pure
from .sinkgd import SinkGD

from lion_pytorch import Lion
from apollo_torch import APOLLOAdamW as APOLLO

from .sage import SAGE
from .sage_universal import UniSAGE

SAGE_lion = partial(SAGE, hybrid=True, lion=True)
SAGE_hybrid = partial(UniSAGE, hybrid=True)
SAGE_pure = partial(UniSAGE, hybrid=False)
