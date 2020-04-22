"""Module provides interaface for used models
"""

from models.fit import FitFcn
from models.sir import sir_step, sihr_step, seir_step
from models.utils import model_iterator, one_minus_logistic_fcn
