"""Tests for SIR model in this repo
* Compares conserved quantities
* Compares model against Penn CHIME w/wo social policies
* Checks logisitc policies in extreme limit
"""
from typing import Tuple
from datetime import date

from pytest import fixture

from numpy import zeros
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal, assert_series_equal

from penn_chime.model.parameters import Parameters, Disposition
from penn_chime.model.sir import Sir, sim_sir, calculate_dispositions, calculate_admits

from models import sir_step, FitFcn, one_minus_logistic_fcn

COLS_TO_COMPARE = ["susceptible", "infected", "recovered", "hospitalized_new"]


@fixture(name="penn_chime_setup")
def fixture_penn_chime_setup() -> Tuple[Parameters, Sir]:
    """Initializes penn_chime parameters and SIR model
    """
    p = Parameters(
        current_hospitalized=69,
        date_first_hospitalized=date(2020, 3, 7),
        doubling_time=None,
        hospitalized=Disposition.create(days=7, rate=0.025),
        icu=Disposition.create(days=9, rate=0.0075),
        infectious_days=14,
        market_share=0.15,
        n_days=100,
        population=3600000,
        recovered=0,
        relative_contact_rate=0.3,
        ventilated=Disposition.create(days=10, rate=0.005),
    )
    return p, Sir(p)


@fixture(name="penn_chime_raw_df_no_beta")
def fixture_penn_chime_raw_df_no_beta(penn_chime_setup) -> DataFrame:
    """Runs penn_chime SIR model for no social policies
    """
    p, simsir = penn_chime_setup

    n_days = simsir.raw_df.day.max() - simsir.raw_df.day.min()
    policies = [(simsir.beta, n_days)]
    raw = sim_sir(
        simsir.susceptible,
        simsir.infected,
        p.recovered,
        simsir.gamma,
        -simsir.i_day,
        policies,
    )
    calculate_dispositions(raw, simsir.rates, market_share=p.market_share)
    calculate_admits(raw, simsir.rates)

    raw_df = DataFrame(raw)

    return raw_df


@fixture(name="sir_data")
def fixture_sir_data(penn_chime_setup, penn_chime_raw_df_no_beta):
    """Provides data for local sir module
    """
    p, simsir = penn_chime_setup
    raw_df = penn_chime_raw_df_no_beta
    day0 = raw_df.iloc[0].fillna(0)

    pars = {
        "beta_i": simsir.beta,
        "gamma_i": simsir.gamma,
        "initial_susceptible": day0.susceptible,
        "initial_infected": day0.infected,
        "initial_hospitalized": day0.hospitalized,
        "initial_recovered": day0.recovered,
        "hospitalization_rate": simsir.rates["hospitalized"] * p.market_share,
    }
    x = {
        "n_iter": raw_df.shape[0],
    }
    return x, pars


@fixture(name="sir_data_w_policy")
def fixture_sir_data_w_policy(penn_chime_setup):
    """Provides data for local sir module with implemented policies
    """
    p, simsir = penn_chime_setup
    raw_df = simsir.raw_df
    day0 = raw_df.iloc[0].fillna(0)

    pars = {
        "beta_i": simsir.beta,
        "gamma_i": simsir.gamma,
        "initial_susceptible": day0.susceptible,
        "initial_infected": day0.infected,
        "initial_hospitalized": day0.hospitalized,
        "initial_recovered": day0.recovered,
        "hospitalization_rate": simsir.rates["hospitalized"] * p.market_share,
    }
    x = {
        "n_iter": raw_df.shape[0],
    }
    return x, pars


def test_conserved_n(sir_data):
    """Checks if S + I + R is conserved for local SIR
    """
    x, pars = sir_data

    n_total = 0
    for key in ["susceptible", "infected", "recovered"]:
        n_total += pars[f"initial_{key}"]

    f = FitFcn(sir_step)
    y = f(x, pars)[["susceptible", "infected", "recovered"]].sum(axis=1) - n_total

    assert_series_equal(y, Series([0.0] * len(y)))


def test_sir_vs_penn_chime_no_policies(penn_chime_raw_df_no_beta, sir_data):
    """Compares local SIR against penn_chime SIR for no social policies
    """
    x, pars = sir_data

    f = FitFcn(sir_step)
    y = f(x, pars)

    assert_frame_equal(
        penn_chime_raw_df_no_beta.rename(columns={"hospitalized": "hospitalized_new"})[
            COLS_TO_COMPARE
        ],
        y[COLS_TO_COMPARE],
    )


def test_sir_vs_penn_chime_w_policies(penn_chime_setup, sir_data_w_policy):
    """Compares local SIR against penn_chime SIR for with social policies
    """
    p, sir = penn_chime_setup
    x, pars = sir_data_w_policy

    policies = sir.gen_policy(p)

    def beta_i_fcn(x_iter, **kwargs):  # pylint: disable=W0613
        out = zeros(len(x_iter))
        ii = 0
        for beta, n_days in policies:
            for _ in range(n_days):
                out[ii] = beta
                ii += 1

        return out

    f = FitFcn(sir_step, beta_i_fcn=beta_i_fcn)
    y = f(x, pars)

    assert_frame_equal(
        sir.raw_df.rename(columns={"hospitalized": "hospitalized_new"})[
            COLS_TO_COMPARE
        ],
        y[COLS_TO_COMPARE],
    )


def test_sir_logistic_policy(penn_chime_setup, sir_data_w_policy):
    """Compares local SIR against penn_chime SIR for with social policies
    where social distancing policies are no implemented as a logistic function
    """
    p, sir = penn_chime_setup
    x, pars = sir_data_w_policy

    policies = sir.gen_policy(p)

    # Set up logisitc function to match policies (Sharp decay)
    pars["beta_i"] = policies[0][0]
    pars["ratio"] = 1 - policies[1][0] / policies[0][0]
    pars["x0"] = policies[0][1] - 0.5
    pars["decay_width"] = 1.0e7

    f = FitFcn(sir_step, beta_i_fcn=one_minus_logistic_fcn)
    y = f(x, pars)

    assert_frame_equal(
        sir.raw_df.rename(columns={"hospitalized": "hospitalized_new"})[
            COLS_TO_COMPARE
        ],
        y[COLS_TO_COMPARE],
    )
