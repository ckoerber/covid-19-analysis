"""Models to analyze COVID-19 data
"""
from typing import Generator, Tuple, TypeVar, Callable, Optional, Dict

from numpy import array, exp

FloatLike = TypeVar("FloatLike")
FloatLikeArray = TypeVar("FloatLikeArray")


def sir_step(
    susceptible: FloatLike,
    infected: FloatLike,
    hospitalized: FloatLike,
    recovered: FloatLike,
    **kwargs
) -> Tuple[FloatLike, FloatLike, FloatLike, FloatLike]:
    """Executes SIR step and patches results such that each component is larger zero.

    Arguments:
        susceptible: Susceptible population
        infected: Infected population
        hospitalized: Hospitalized population
        recovered: Recovered population
        kwargs: FloatLike
            beta_i: Growth rate for infected
            gamma_i: Recovery rate for infected
            hospitalization_rate: Percent of new cases becoming hospitalized

    Returns:
        S, I, H, R: FloatLike
            Hospitalized is computed by growth from
            `(r(t) + i(t) - r(t-1) - i(t-1)) * hospitalization_rate`
    """
    total = susceptible + infected + recovered

    is_grow = kwargs["beta_i"] * susceptible * infected
    ir_loss = kwargs["gamma_i"] * infected

    susceptible -= is_grow
    infected += is_grow - ir_loss
    recovered += ir_loss

    susceptible = max(susceptible, 0)
    infected = max(infected, 0)
    recovered = max(recovered, 0)

    rescale = total / (susceptible + infected + recovered)

    susceptible *= rescale
    infected *= rescale
    recovered *= rescale

    hospitalized = is_grow
    hospitalized *= kwargs["hospitalization_rate"]

    return (
        susceptible,
        infected,
        hospitalized,
        recovered,
    )


def sihr_step(  # pylint: disable=R0913
    susceptible: FloatLike,
    infected: FloatLike,
    hospitalized: FloatLike,
    recovered: FloatLike,
    **kwargs
) -> Tuple[FloatLike, FloatLike, FloatLike, FloatLike]:
    """Executes SIHR step and patches results such that each component is larger zero.

    Arguments:
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

    Returns:
        S, I, H, R: FloatLike
    """
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

    susceptible = max(susceptible, 0)
    infected = max(infected, 0)
    hospitalized = max(hospitalized, 0)
    recovered = max(recovered, 0)

    rescale = total / (susceptible + infected + recovered + hospitalized)

    return (
        susceptible * rescale,
        infected * rescale,
        hospitalized * rescale,
        recovered * rescale,
    )


def model_iterator(
    n_iter: int,
    sir_fcn: Callable,
    *args,
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
    pars = kwargs.copy()
    beta_i_schedule = (
        beta_i_fcn(n_iter, **kwargs)
        if beta_i_fcn is not None
        else [kwargs.get("beta_i", None)] * n_iter
    )

    for beta_i in beta_i_schedule:
        yield args
        pars["beta_i"] = beta_i
        args = sir_fcn(*args, **pars)


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

    _columns = ("susceptible", "infected", "hospitalized", "recovered")

    def __init__(
        self,
        sir_fcn: Callable,
        beta_i_fcn: Optional[Callable] = None,
        columns: Tuple[str] = ("infected", "hospitalized"),
    ):
        """Initializes fit function

        Arguments:
            sir_fcn: The function which executes a SIR step
            beta_i_fcn: Function which generates a schedule for the growth rate
            columns: Function call return specified columns as arrays
        """
        self.col_index = []
        for col in columns:
            if not col in self._columns:
                raise KeyError(
                    "Key {col} not in expected columns {cols}".format(
                        col=col, cols=self._columns
                    )
                )
            for i, icol in enumerate(self._columns):
                if col == icol:
                    self.col_index.append(i)
                    break

        self.columns = columns
        self.sir_fcn = sir_fcn
        self.beta_i_fcn = beta_i_fcn

    def __call__(
        self, x: Dict[str, FloatLike], p: Dict[str, FloatLike]
    ) -> FloatLikeArray:
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

        kwargs = p.copy()
        args = (
            s if s is not None else kwargs.pop("initial_susceptible"),
            i if i is not None else kwargs.pop("initial_infected"),
            h if h is not None else kwargs.pop("initial_hospitalized"),
            r if r is not None else kwargs.pop("initial_recovered"),
        )
        y = array(
            list(
                model_iterator(
                    x["n_iter"],
                    self.sir_fcn,
                    *args,
                    beta_i_fcn=self.beta_i_fcn,
                    **kwargs
                )
            )
        )[:, self.col_index]
        return y if len(self.col_index) > 1 else y.flatten()
