"""Models to analyze COVID-19 data
"""
from typing import Generator, Tuple, TypeVar, Callable, Optional, Dict, List, Union

from numpy import exp
from pandas import DataFrame

FloatLike = TypeVar("FloatLike")
FloatLikeArray = TypeVar("FloatLikeArray")


def sir_step(sir: Dict[str, FloatLike], **kwargs) -> Dict[str, FloatLike]:
    """Executes SIR step and patches results such that each component is larger zero.

    Arguments:
        sir: FloatLike
            susceptible: Susceptible population
            infected: Infected population
            recovered: Recovered population
        kwargs: FloatLike
            beta_i: Growth rate for infected
            gamma_i: Recovery rate for infected
            hospitalization_rate: Percent of new cases becoming hospitalized
    """
    susceptible = sir["susceptible"]
    infected = sir["infected"]
    recovered = sir["recovered"]

    total = susceptible + infected + recovered

    is_grow = kwargs["beta_i"] * susceptible * infected
    ir_loss = kwargs["gamma_i"] * infected

    susceptible -= is_grow
    infected += is_grow - ir_loss
    recovered += ir_loss

    out = {
        "infected_new": is_grow,
        "recovered_new": ir_loss,
    }

    susceptible = max(susceptible, 0)
    infected = max(infected, 0)
    recovered = max(recovered, 0)

    rescale = total / (susceptible + infected + recovered)

    out["susceptible"] = susceptible * rescale
    out["infected"] = infected * rescale
    out["recovered"] = recovered * rescale

    out["hospitalized_new"] = is_grow * kwargs["hospitalization_rate"]

    return out


def sihr_step(  # pylint: disable=R0913
    sihr: Dict[str, FloatLike], **kwargs
) -> Dict[str, FloatLike]:
    """Executes SIHR step and patches results such that each component is larger zero.

    Arguments:
        sihr: FloatLike
            susceptible: Susceptible population
            infected: Infected population
            hospitalized: Hospitalized population
            recovered: Recovered population
        kwargs: FloatLike
            beta_i: Growth rate for infected
            gamma_i: Recovery rate for infected
            beta_h: Growth rate for hospitalized
            gamma_i: Recovery rate for hospitalized
            capacity: Maximal number of hospitalized population
    """
    susceptible = sihr["susceptible"]
    infected = sihr["susceptible"]
    hospitalized = sihr["hospitalized"]
    recovered = sihr["infected"]

    total = susceptible + infected + recovered + hospitalized

    is_grow = kwargs["beta_i"] * susceptible * infected
    ir_loss = kwargs["gamma_i"] * infected
    ih_loss = min(
        kwargs["beta_h"] * infected, max(kwargs["capacity"] - hospitalized, 0)
    )
    hr_loss = kwargs["gamma_h"] * hospitalized

    susceptible -= is_grow
    infected += is_grow - ih_loss - ir_loss
    hospitalized += ih_loss - hr_loss
    recovered += ir_loss + hr_loss

    out = {
        "infected_new": is_grow,
        "hospitalized_new": ih_loss,
        "recovered_new": ir_loss + hr_loss,
    }

    susceptible = max(susceptible, 0)
    infected = max(infected, 0)
    hospitalized = max(hospitalized, 0)
    recovered = max(recovered, 0)

    rescale = total / (susceptible + infected + recovered + hospitalized)

    out["susceptible"] = susceptible * rescale
    out["infected"] = infected * rescale
    out["hospitalized"] = hospitalized * rescale
    out["recovered"] = recovered * rescale

    return out


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
    beta_i_schedule = (
        beta_i_fcn(n_iter, **kwargs)
        if beta_i_fcn is not None
        else [kwargs.get("beta_i", None)] * n_iter
    )

    for beta_i in beta_i_schedule:
        yield data
        pars["beta_i"] = beta_i
        data = sir_fcn(data, **pars)


def one_minus_logistic_fcn(  # pylint: disable=C0103
    x: FloatLikeArray,
    amplitude: FloatLike = 1.0,
    decay_width: FloatLike = 1.0,
    x0: FloatLike = 0.0,
) -> FloatLikeArray:
    """Computes `1 - A / (1 + exp(-w(x-x0)))`.
    """
    return 1 - amplitude / (1 + exp(-decay_width * (x - x0)))


class FitFcn:  # pylint: disable=R0903
    """Fit function wrapper
    """

    def __init__(
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
            columns: Function call return specified columns as arrays
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
