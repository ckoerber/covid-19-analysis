"""Different implementations of SIR models to analyze COVID-19 data
"""
from typing import Dict

from models.utils import FloatLike


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
    infected = sihr["infected"]
    hospitalized = sihr["hospitalized"]
    recovered = sihr["infected"]

    total = susceptible + infected + recovered + hospitalized

    is_grow = kwargs["beta_i"] * susceptible * infected
    ir_loss = kwargs["gamma_i"] * infected
    ih_loss = kwargs["beta_h"] * infected
    open_h = kwargs["capacity"] - hospitalized
    if ih_loss > open_h:  # take this notation to conserve types
        ih_loss -= ih_loss - open_h
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
