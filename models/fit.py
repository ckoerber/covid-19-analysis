"""Interface for models to simplify fitting
"""
from typing import Dict, Optional, List, Callable, Union, Tuple

from datetime import timedelta

from numpy import log, delete
from pandas import DataFrame

from models.utils import FloatLike, FloatLikeArray, model_iterator


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
        self.drop_rows = drop_rows or []

    @staticmethod
    def convert_kwargs(
        x: Dict[str, FloatLike], pars: Dict[str, FloatLike]
    ) -> Tuple[Dict[str, FloatLike], Dict[str, FloatLike]]:
        """Tries to run conversions on prior parameters
        """
        kwargs = dict(pars)
        xx = dict(x)

        if "date" in xx:
            bin_size = int(xx["date"].freq / timedelta(days=1))
            if "bin_size" in xx:
                assert bin_size == xx["bin_size"]
            else:
                xx["bin_size"] = bin_size

            n_iter = len(xx["date"])
            if "n_iter" in xx:
                assert n_iter == xx["n_iter"]
            else:
                xx["n_iter"] = n_iter

        recovery_days_i = kwargs.pop("recovery_days_i", None)
        if recovery_days_i is not None:
            kwargs["gamma_i"] = 1 / (recovery_days_i / xx["bin_size"])

        recovery_days_h = kwargs.pop("recovery_days_h", None)
        if recovery_days_h is not None:
            kwargs["gamma_h"] = 1 / (recovery_days_h / xx["bin_size"])

        social_distance_delay = kwargs.pop("social_distance_delay", None)
        if social_distance_delay is not None:
            kwargs["x0"] = social_distance_delay / xx["bin_size"]

        social_distance_halfing_days = kwargs.pop("social_distance_halfing_days", None)
        if social_distance_halfing_days is not None:
            kwargs["decay_width"] = 1 / (social_distance_halfing_days / xx["bin_size"])

        inital_doubling_time = kwargs.pop("inital_doubling_time", None)
        if inital_doubling_time is not None:
            kwargs["beta_i"] = (
                log(2) / (inital_doubling_time / xx["bin_size"]) + kwargs["gamma_i"]
            ) / xx["initial_susceptible"]

        return xx, kwargs

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

        xx, kwargs = self.convert_kwargs(x, p)
        kwargs["capacity"] = x.get("capacity", None)

        data = {
            "susceptible": s if s is not None else kwargs["initial_susceptible"],
            "infected": i if i is not None else kwargs["initial_infected"],
            "hospitalized": h if h is not None else kwargs["initial_hospitalized"],
            "recovered": r if r is not None else kwargs["initial_recovered"],
        }
        df = DataFrame(
            data=model_iterator(
                xx["n_iter"], self.sir_fcn, data, beta_i_fcn=self.beta_i_fcn, **kwargs
            )
        )

        return self.post_process(df, xx, kwargs)

    def post_process(
        self, df: DataFrame, xx: Dict[str, FloatLike], kwargs: Dict[str, FloatLike]
    ) -> Union[DataFrame, FloatLikeArray]:
        """Runs post processing steps on computed SIR-like dataframe
        """
        if (
            self.sir_fcn.__name__ == "sir_step"
            and "length_of_stay" in xx
            and (not self.columns or "hospitalized" in self.columns)
        ):
            initial_hospitalized = (
                xx["initial_hospitalized"]
                if "initial_hospitalized" in xx
                else kwargs["initial_hospitalized"]
            )
            shift = xx["length_of_stay"] // (xx["bin_size"] if "bin_size" in xx else 1)
            df["hospitalized"] = df.hospitalized_new.cumsum()
            df["hospitalized"][0] = initial_hospitalized
            df["hospitalized"] -= df["hospitalized"].shift(shift).fillna(0)

        if self.drop_rows:
            df = df.drop(index=self.drop_rows)
        if self.columns:
            df = df[self.columns]
        if self.as_array:
            df = df.values
            if len(df.shape) == 1 or df.shape[1] == 1:
                df = df.flatten()
        else:
            if "date" in xx:
                df.index = delete(xx["date"], self.drop_rows)

        return df
