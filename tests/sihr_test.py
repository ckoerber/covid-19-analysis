"""Tests for SIHR model in this repo
* Compares conserved quantities
* Compares model against SIR w/wo social policies in limit of no hospitalizations
"""
from typing import Tuple
from datetime import date

from pytest import fixture

from numpy import zeros, inf
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal, assert_series_equal

from models import sir_step, FitFcn, sihr_step

from tests.sir_test import (
    fixture_sir_data,
    fixture_penn_chime_setup,
    fixture_penn_chime_raw_df_no_beta,
)

COLS_TO_COMPARE = ["susceptible", "infected", "recovered", "hospitalized_new"]


@fixture(name="sihr_data")
def fixture_sihr_data(sir_data):
    """Returns data for the SIHR model
    """
    x, p = sir_data
    pp = p.copy()
    xx = x.copy()

    xx["capacity"] = inf
    pp.pop("hospitalization_rate")
    pp["gamma_h"] = p["gamma_i"]
    pp["beta_h"] = 0.01

    return xx, pp


def test_conserved_n(sihr_data):
    """Checks if S + I + H + R is conserved for local SIHR
    """
    x, pars = sihr_data

    cols = ["susceptible", "infected", "hospitalized", "recovered"]

    n_total = 0
    for key in cols:
        n_total += pars[f"initial_{key}"]

    f = FitFcn(sihr_step)
    y = f(x, pars)[cols].sum(axis=1) - n_total

    assert_series_equal(y, Series([0.0] * len(y)))


def test_compare_sir_vs_sihr(sir_data, sihr_data):
    """Checks if SIHR and SIR return same results if no hospitalizations
    """
    x_sir, pars_sir = sir_data
    x_sihr, pars_sihr = sihr_data
    pars_sihr["beta_h"] = 0.0  # no hospitalizations

    cols = ["susceptible", "infected", "recovered"]

    f_sir = FitFcn(sir_step, columns=cols)
    f_sihr = FitFcn(sihr_step, columns=cols)

    df_sir = f_sir(x_sir, pars_sir)
    df_sihr = f_sihr(x_sihr, pars_sihr)

    assert_frame_equal(df_sir, df_sihr)


def test_capacity_limit(sihr_data):
    """Runs SIHR with finite capacity to ensure H does not surpass limit
    """
    x, pars = sihr_data

    capacity_limit = 1000

    f = FitFcn(sihr_step, columns=["hospitalized"], drop_rows=[0], as_array=True)

    # Check that virtual limit is surpassed if no boundary passed to simulation
    x["capacity"] = inf
    y = f(x, pars)
    assert any(y > capacity_limit)

    # Now implement limit and check it holds
    x["capacity"] = capacity_limit
    y = f(x, pars)
    assert all(y.astype(int) <= capacity_limit)
