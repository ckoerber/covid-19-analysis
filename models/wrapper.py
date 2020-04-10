"""Wrapper module to simplify fitting interface
"""
from typing import Dict, Optional, Tuple, Callable, TypeVar

from abc import ABC

from pandas import DataFrame
from pandas.tseries.offsets import Day
from pandas.core.indexes.datetimes import DatetimeIndex

from numpy import log
from gvar import gvar
from lsqfit import nonlinear_fit

from models.utils import FloatLike, FloatLikeArray, one_minus_logistic_fcn
from models.fit import FitFcn
from models.sir import sir_step, sihr_step

Fit = TypeVar("Fit")


class Model(ABC):
    """Abstract interaface for initializing models

    The primary purpose of this class is setting default priors
    """

    _prior: Dict[str, FloatLike] = {}
    _xx: Dict[str, FloatLike] = {
        "initial_susceptible": int(1.8e6),
        "initial_recovered": 0,
    }
    _yy: FloatLikeArray = None
    _columns: Tuple[str] = None
    _fcn: Callable = None
    verbose_name: Optional[str] = None

    def __init__(
        self,
        data: DataFrame,
        columns: Tuple[str] = ("infected_new",),
        xx: Optional[Dict[str, FloatLike]] = None,
        prior: Optional[Dict[str, FloatLike]] = None,
    ):
        """Initializes model interface

        Arguments:
            data: DataFrame for fit data. Columns must be
             `["hospitalized_cummulative", "hospitalized_new", "infected_new"]`
              and index must be `date` of type `DatetimeIndex` with `freq` `n*Day`
            columns: Tuple[str] = ("infected_new",),
                Data to fit.
            xx: Optional[Dict[str, FloatLike]] = None,
            prior: Optional[Dict[str, FloatLike]] = None,
        """
        self._columns = columns

        assert data.index.name == "date"
        self._data = data.sort_index(ascending=True)

        assert isinstance(self.data.index, DatetimeIndex)
        assert isinstance(self.data.index.freq, Day)
        self.bin_size = self.data.index.freq.n

        for col in ["hospitalized_cummulative", "hospitalized_new", "infected_new"]:
            assert col in self.data.columns

        xx = xx or {
            "initial_hospitalized": self.data.hospitalized_cummulative.iloc[0],
            "initial_recovered": 0,
            "n_iter": self.data.shape[0] + 1,
        }
        self._xx.update(xx)

        prior = prior or {}
        self._prior.update(prior)

        self.set_y()

    def set_y(self, **kwargs: Tuple[float, float]):
        """Sets y values for input data.

        Adds non-correlated uncertainty to prioritize fit regions.

        Arguments:
            kwargs: Tuple[float, float]
                `column_name=(y_sdev_min, y_sdev_fact)`
                with y_sdev = max(y_sdev_min, y*y_sdev_fact)
        """
        y_means = []
        y_sdevs = []
        for col in self.columns:
            default = (10, 0.05) if col == "hospitalized_new" else (100, 0.5)
            y_min, y_fact = kwargs.get(col, default)

            y_mean = self.data[col].values
            y_means.append(y_mean)

            y_sdev = [max(yi * y_fact, y_min) for yi in y_mean]
            y_sdevs.append(y_sdev)

        self._yy = gvar(y_means, y_sdevs).T

    @property
    def prior(self) -> Dict[str, FloatLike]:
        """Returns copy of prior
        """
        return self._prior.copy()

    @property
    def xx(self) -> Dict[str, FloatLike]:  # pylint: disable=C0103
        """Returns copy of input data
        """
        return self._xx.copy()

    @property
    def yy(self) -> FloatLikeArray:  # pylint: disable=C0103
        """Returns copy of input data
        """
        return self._yy

    @property
    def fcn(self) -> Callable:
        """Returns fit function
        """
        return self._fcn

    @property
    def columns(self) -> Tuple[str]:  # pylint: disable=C0103
        """Returns columns to be fitted
        """
        return self._columns

    @property
    def data(self) -> DataFrame:  # pylint: disable=C0103
        """Returns data
        """
        return self._data

    def fit(  # pylint: disable=C0103
        self,
        xx: Optional[Dict[str, FloatLike]] = None,
        prior: Optional[Dict[str, FloatLike]] = None,
        yy: Optional[FloatLikeArray] = None,
    ) -> Fit:
        """Runs fit for default or given data
        """
        xx = xx or self.xx
        prior = prior or self.prior
        yy = yy or self.yy
        return nonlinear_fit(data=(xx, yy), prior=prior, fcn=self.fcn)


class SIRFixedBeta(Model):
    verbose_name = "SIR model with fixed beta"

    def __init__(
        self,
        data: DataFrame,
        columns: Tuple[str] = ("infected_new",),
        xx: Optional[Dict[str, FloatLike]] = None,
        prior: Optional[Dict[str, FloatLike]] = None,
    ):
        """
        """
        super().__init__(data, columns=columns, xx=xx, prior=prior)

        self.fcn = FitFcn(sir_step, columns=self.columns, as_array=True, drop_rows=[0])

        beta0 = log(2) / 2 / self.xx["initial_susceptible"]
        prior = prior or {
            "beta_i": gvar(beta0, beta0 * 0.8) / self.bin_size,
            "gamma_i": 1 / (gvar(14, 5) / self.bin_size),
            "initial_infected": gvar(1.0e4, 2.0e4),
            "hospitalization_rate": gvar(0.05, 0.1),
        }
        self.prior.update(prior)


class SIRLogisticBeta(Model):
    verbose_name = "SIR model with beta(t) = logistic"

    def __init__(
        self,
        data: DataFrame,
        columns: Tuple[str] = ("infected_new",),
        xx: Optional[Dict[str, FloatLike]] = None,
        prior: Optional[Dict[str, FloatLike]] = None,
    ):
        """
        """
        super().__init__(data, columns=columns, xx=xx, prior=prior)

        self.fcn = FitFcn(
            sir_step,
            columns=self.columns,
            as_array=True,
            drop_rows=[0],
            beta_i_fcn=one_minus_logistic_fcn,
        )

        beta0 = log(2) / 2 / self.xx["initial_susceptible"]
        prior = prior or {
            "beta_i": gvar(beta0, beta0 * 0.8) / self.bin_size,
            "gamma_i": 1 / (gvar(14, 5) / self.bin_size),
            "initial_infected": gvar(1.0e4, 2.0e4),
            "hospitalization_rate": gvar(0.05, 0.1),
            "ratio": gvar(0.7, 0.3),
            "decay_width": 1 / gvar(4, 1),
            "x0": gvar(9, 2),
        }
        self.prior.update(prior)


class SIHRFixedBeta(Model):
    verbose_name = "SIHR model with fixed beta"

    def __init__(
        self,
        data: DataFrame,
        columns: Tuple[str] = ("hospitalized_new",),
        xx: Optional[Dict[str, FloatLike]] = None,
        prior: Optional[Dict[str, FloatLike]] = None,
    ):
        """
        """
        super().__init__(data, columns=columns, xx=xx, prior=prior)

        self.fcn = FitFcn(
            sihr_step, columns=self.columns, as_array=True, drop_rows=[0],
        )
        beta0 = log(2) / 2 / self.xx["initial_susceptible"]
        prior = prior or {
            "beta_i": gvar(beta0, beta0 * 0.8) / self.bin_size,
            "gamma_i": 1 / (gvar(14, 5) / self.bin_size),
            "initial_infected": gvar(1.0e4, 2.0e4),
            "beta_h": gvar(0.05, 0.1),
            "gamma_h": 1 / (gvar(14, 5) / self.bin_size),
        }
        self.prior.update(prior)


class SIHRLogisticBeta(Model):
    verbose_name = "SIHR model with beta(t) = logistic"

    def __init__(
        self,
        data: DataFrame,
        columns: Tuple[str] = ("hospitalized_new",),
        xx: Optional[Dict[str, FloatLike]] = None,
        prior: Optional[Dict[str, FloatLike]] = None,
    ):
        """
        """
        super().__init__(data, columns=columns, xx=xx, prior=prior)

        self.fcn = FitFcn(
            sihr_step,
            columns=self.columns,
            as_array=True,
            drop_rows=[0],
            beta_i_fcn=one_minus_logistic_fcn,
        )

        beta0 = log(2) / 2 / self.xx["initial_susceptible"]
        prior = prior or {
            "beta_i": gvar(beta0, beta0 * 0.8) / self.bin_size,
            "gamma_i": 1 / (gvar(14, 5) / self.bin_size),
            "initial_infected": gvar(1.0e4, 2.0e4),
            "beta_h": gvar(0.05, 0.1),
            "gamma_h": 1 / (gvar(14, 5) / self.bin_size),
            "ratio": gvar(0.7, 0.3),
            "decay_width": 1 / gvar(4, 1),
            "x0": gvar(9, 2),
        }
        self.prior.update(prior)
