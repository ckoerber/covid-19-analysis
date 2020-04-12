"""Interface for models to simplify fitting
"""
from typing import Dict, Optional, List, Callable, Union

from numpy import log
from pandas import DataFrame

from models.utils import FloatLike, FloatLikeArray, model_iterator, get_doubling_time


class FitFcn:  # pylint: disable=R0903
    """Fit function wrapper
    """

    def __init__(  # pylint: disable=R0913
        self,
        sir_fcn: Callable,
        beta_i_fcn: Optional[Callable] = None,
        columns: Optional[List[str]] = None,
        as_array: bool = False,
        drop_rows: Optional[List[int]] = None,
    ):
        """Initializes fit function

        Arguments:
            sir_fcn: The function which executes a SIR step
            beta_i_fcn: Function which generates a schedule for the growth rate
            columns: Function call return specified columns only
            as_array: If true, function call returns array. Else DataFrame.
            drop_rows: Drop selected rows from function call.
        """
        self.sir_fcn = sir_fcn
        self.beta_i_fcn = beta_i_fcn
        self.columns = columns
        self.as_array = as_array
        self.drop_rows = drop_rows

    def __call__(
        self, x: Dict[str, FloatLike], p: Dict[str, FloatLike]
    ) -> Union[DataFrame, FloatLikeArray]:
        """Runs SIR fcn step for input values x and p

        Either x or p must contain keys
            * initial_susceptible
            * initial_infected
            * initial_hospitalized
            * initial_recovered

        Arguments:
            x: Must contain key "n_iter" (number of iterations)
            p: Parameters for SIR model and beta schedule if specified
        """
        s = x.get("initial_susceptible", None)
        i = x.get("initial_infected", None)
        h = x.get("initial_hospitalized", None)
        r = x.get("initial_recovered", None)

        kwargs = dict(p)
        kwargs["capacity"] = x.get("capacity", None)

        recovery_days = kwargs.pop("recovery_days", None)
        if recovery_days is not None:
            kwargs["gamma_i"] = 1 / (recovery_days / x["bin_size"])
            kwargs["gamma_h"] = 1 / (recovery_days / x["bin_size"])

        inital_doubling_time = kwargs.pop("inital_doubling_time", None)
        if inital_doubling_time is not None:
            kwargs["beta_i"] = (
                log(2) / (inital_doubling_time / x["bin_size"]) + kwargs["gamma_i"]
            ) / s

        data = {
            "susceptible": s if s is not None else kwargs.pop("initial_susceptible"),
            "infected": i if i is not None else kwargs.pop("initial_infected"),
            "hospitalized": h if h is not None else kwargs.pop("initial_hospitalized"),
            "recovered": r if r is not None else kwargs.pop("initial_recovered"),
        }
        out = DataFrame(
            data=model_iterator(
                x["n_iter"], self.sir_fcn, data, beta_i_fcn=self.beta_i_fcn, **kwargs
            )
        )

        if self.drop_rows:
            out = out.drop(index=self.drop_rows)
        if self.columns:
            out = out[self.columns]
        if self.as_array:
            out = out.values
            if len(out.shape) == 1 or out.shape[1] == 1:
                out = out.flatten()
        return out
