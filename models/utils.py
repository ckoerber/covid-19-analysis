"""Utility functions for the SIR models
"""
from typing import Generator, Tuple, TypeVar, Callable, Optional, Dict

from numpy import exp, arange, log

FloatLike = TypeVar("FloatLike")
FloatLikeArray = TypeVar("FloatLikeArray")


def model_iterator(
    n_iter: int,
    sir_fcn: Callable,
    data: Dict[str, FloatLike],
    beta_i_fcn: Optional[Callable] = None,
    **kwargs
) -> Generator[Tuple[FloatLike, FloatLike, FloatLike, FloatLike], None, None]:
    """Iterates model to build up SIR data

    Initial data is at day zero (no step).

    Arguments:
        n_iter: Number of iterations
        sir_fcn: The SIR model step function
        beta_i_fcn: Function which maps infected growth for given kwargs
        kwargs: Parameters to consturct beta_i schedule and sir step
    """
    pars = dict(kwargs)
    x = arange(n_iter)
    beta_i_schedule = (
        beta_i_fcn(x, amplitude=kwargs["beta_i"], **kwargs)
        if beta_i_fcn is not None
        else [kwargs.get("beta_i", None)] * n_iter
    )

    for beta_i in beta_i_schedule:
        yield data
        pars["beta_i"] = beta_i
        data = sir_fcn(data, **pars)


def one_minus_logistic_fcn(  # pylint: disable=C0103, W0613
    x: FloatLikeArray,
    ratio: FloatLike,
    decay_width: FloatLike,
    x0: FloatLike,
    amplitude: FloatLike = 1.0,
    **kwargs
) -> FloatLikeArray:
    """Computes `A(1 - r / (1 + exp(-w(x-x0)))`.
    """
    return amplitude * (1 - ratio / (1 + exp(-decay_width * (x - x0))))


def get_doubling_time(
    beta_i: FloatLike, gamma_i: FloatLike, susceptible: FloatLike
) -> FloatLike:
    """Converts beta_i, gamma_i and susceptible to doubling time.
    """
    return log(2) / (beta_i * susceptible - gamma_i)
