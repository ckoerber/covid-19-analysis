"""Routines to pull data frames and update columns

Repo:

https://github.com/nychealth/coronavirus-data
"""
from typing import Optional, Tuple
from datetime import date, timedelta

from pandas import read_csv, DataFrame

BASE_URL = "https://raw.githubusercontent.com/nychealth/coronavirus-data/"

COMMIT_HASH_LAST = "e19db289166f73282d39dfcef0d47a324d654c07"


def prepare_case_hosp_death(
    commit_hash: str = "master",
    drop_days_end: int = 3,
    bin_day_range: Optional[int] = 3,
) -> Tuple[DataFrame, DataFrame]:
    """Downloads `case_hosp_death` from source and prepares columns

    Arguments:
        commit_hash: Hash of the commit to use to pull data from
        drop_days_end: Leaves out rows at the end of the data frame to avoid errors
            related to delayed reporting
        bin_day_range: Averages resluts over several days

    Returns:
        total cases/hospitalized/death and new cases/hospitalized/death frames
    """
    file_name = "case-hosp-death.csv"
    url = "/".join([BASE_URL, commit_hash, file_name])

    days_ago = date.today() - timedelta(days=drop_days_end)

    nyc_chp_df = (
        read_csv(url, parse_dates=["DATE_OF_INTEREST"])
        .fillna(0)
        .astype(
            dtype={
                "NEW_COVID_CASE_COUNT": int,
                "HOSPITALIZED_CASE_COUNT": int,
                "DEATH_COUNT": int,
            },
        )
        .rename(
            columns={
                "DATE_OF_INTEREST": "date",
                "NEW_COVID_CASE_COUNT": "infected_new",
                "HOSPITALIZED_CASE_COUNT": "hospitalized_new",
                "DEATH_COUNT": "death_new",
            }
        )
        .set_index("date")
        .loc[:days_ago]
    )

    if bin_day_range:
        nyc_chp_df = nyc_chp_df.resample(f"{bin_day_range}D").sum()

    for col in nyc_chp_df.columns:
        if "new" in col:
            nyc_chp_df[col.replace("new", "cummulative")] = nyc_chp_df[col].cumsum()

    return nyc_chp_df[sorted(nyc_chp_df.columns)]


def prepare_testing(
    commit_hash: str = "master", drop_days: int = 3, bin_day_range: int = 0,
) -> Tuple[DataFrame, DataFrame]:
    """Downloads `case_hosp_death` from source and prepares columns

    Arguments:
        commit_hash: Hash of the commit to use to pull data from
        drop_days: Drop days where the extraction is in a `drop_days` range from the
            specimen date
        bin_day_range: Averages resluts over several days

    Returns:
        total cases/hospitalized/death and new cases/hospitalized/death frames
    """
    file_name = "testing.csv"
    url = "/".join([BASE_URL, commit_hash, file_name])

    time_delta = timedelta(days=drop_days)

    df = (
        read_csv(url, parse_dates=["extract_date", "specimen_date"])
        .rename(
            columns={
                "Number_tested": "tested_new",
                "Number_confirmed": "infected_new",
                "Number_hospitalized": "hospitalized_new",
                "Number_deaths": "death_new",
            }
        )
        .set_index("specimen_date")
    )
    df = df[df["extract_date"] > df.index + time_delta]

    df = df.drop(columns=["extract_date"])

    if bin_day_range:
        df = df.resample(f"{bin_day_range}D").sum()

    for col in df.columns:
        if "new" in col:
            df[col.replace("new", "cummulative")] = df[col].cumsum()

    return df[sorted(df.columns)]
