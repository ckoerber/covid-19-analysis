"""Functions to simplify plotting
"""
from typing import List, Optional, Dict

from copy import deepcopy
from datetime import date

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from numpy import arange
from lsqfit import nonlinear_fit
from gvar import fmt_values, fmt_errorbudget, mean, sdev

from pandas import DataFrame


def hex_to_rgba(hex_str: str, alpha: float):
    """Converts hex string to rgba
    """
    if "#" in hex_str:
        hex_str = hex_str.replace("#", "")
    out = "rgba({0},{1},{2},{alpha})".format(
        *[int(hex_str[i : i + 2], 16) for i in (0, 2, 4)], alpha=alpha
    )
    return out


def add_gvar_scatter(
    fig: go.Figure,
    gv_mode: str = "scatter",
    y_min: float = 0,
    y_max: Optional[float] = None,
    **kwargs,
) -> go.Figure:
    """Wraps adds scatter for gvars
    """
    x = kwargs.pop("x")
    y = kwargs.pop("y")

    y_mean = mean(y)  # pylint: disable=E1101
    y_sdev = sdev(y)  # pylint: disable=E1101

    row, col = kwargs.pop("row", None), kwargs.pop("col", None)
    color = kwargs.pop("color", None)

    yy_min = [max(yy, y_min) for yy in y_mean - y_sdev]
    yy_max = [min(yy, y_max or yy) for yy in y_mean + y_sdev]

    if gv_mode == "band":
        kwargs.pop("mode", None)
        name = kwargs.pop("name", None)
        showlegend = kwargs.pop("showlegend", True)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=yy_min,
                fill=None,
                mode="lines",
                showlegend=False,
                line={"color": color, "shape": "spline"},
                **kwargs,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=yy_max,
                fill="tonexty",
                mode="lines",
                name=name,
                line={"color": color, "shape": "spline"},
                fillcolor=hex_to_rgba(color, 0.5),
                showlegend=showlegend,
                **kwargs,
            ),
            row=row,
            col=col,
        )
    elif gv_mode == "scatter":
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_mean.astype(int),
                error_y_array=y_sdev,
                **kwargs,
                line={"color": color},
            ),
            row=row,
            col=col,
        )
    else:
        raise KeyError("gv_mode: Only band and scatter are implemented.")

    return fig


def plot_sir_sihr_comparison(df_sir: DataFrame, df_sihr: DataFrame, capacity: int):
    """Multiframe plot comparing SIR vs SIHR
    """

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "New hospitalizations",
            "New Infections",
            "Hospitalized",
            "Infected (inclusive hospitalized)",
        ),
        shared_xaxes=True,
        x_title="Days",
    )

    fig.add_trace(
        go.Scatter(
            x=df_sir.index,
            y=df_sir.hospitalized_new,
            name="SIR",
            mode="markers+lines",
            line_color="#1f77b4",
        ),
        row=1,
        col=1,
    )

    add_gvar_scatter(
        fig,
        x=df_sihr.index,
        y=df_sihr.hospitalized_new.values,
        name="SIHR",
        gv_mode="band",
        row=1,
        col=1,
        color="#bcbd22",
    )

    fig.add_trace(
        go.Scatter(
            x=df_sir.index,
            y=df_sir.infected_new,
            mode="markers+lines",
            line_color="#1f77b4",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    add_gvar_scatter(
        fig,
        x=df_sihr.index,
        y=df_sihr.infected_new.values,
        gv_mode="band",
        row=1,
        col=2,
        color="#bcbd22",
        showlegend=False,
    )

    add_gvar_scatter(
        fig,
        x=df_sihr.index,
        y=df_sihr.hospitalized.values,
        gv_mode="band",
        row=2,
        col=1,
        color="#bcbd22",
        showlegend=False,
    )
    if any(
        capacity * 0.9
        < mean(df_sihr.hospitalized.values) + sdev(df_sihr.hospitalized.values)
    ):
        fig.add_trace(
            go.Scatter(
                x=df_sir.index,
                y=[capacity] * df_sir.shape[0],
                mode="lines",
                line_color="black",
                name="Hospital capacity",
            ),
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=df_sir.index,
            y=df_sir.infected,
            mode="markers+lines",
            line_color="#1f77b4",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    add_gvar_scatter(
        fig,
        x=df_sihr.index,
        y=df_sihr.infected_inclusive.values,
        gv_mode="band",
        row=2,
        col=2,
        color="#bcbd22",
        showlegend=False,
    )

    fig.update_layout(
        title="Comparison SIR vs SIHR (Capacity={capacity})".format(capacity=capacity)
    )

    return fig


COLUMN_NAME_MAP = {
    "initial_infected": "I(0)",
    "hospitalization_rate": "P_H",
    "inital_doubling_time": "t2(0)",
    "recovery_days_i": "Recovery I",
    "recovery_days_h": "Recovery H",
    "ratio": "R",
    "social_distance_halfing_days": "w sd",
    "social_distance_delay": "dt sd",
}


def summarize_fit(fit: nonlinear_fit):
    """Summarizes a non linear fit object
    """
    outputs = dict(fit.p)
    inputs = dict(Data=fit.y, **fit.prior)

    for key, val in COLUMN_NAME_MAP.items():
        if key in outputs:
            outputs[val] = outputs.pop(key)
        if key in inputs:
            inputs[val] = inputs.pop(key)

    print("Legend:")
    print(
        "\n".join(
            f"- {val}: {key}" for key, val in COLUMN_NAME_MAP.items() if key in fit.p
        ),
        "\n",
    )

    print(
        fit,
        "\n-------",
        "\nResult:",
        "\n-------",
        "\n\n",
        fmt_values(outputs),
        "\n",
        "\n-------------",
        "\nError budget:",
        "\n-------------",
        "\n\n",
        fmt_errorbudget(outputs, inputs, percent=False),
    )


def plot_fits(fit: nonlinear_fit, x: Optional[List[date]] = None) -> go.Figure:
    """Plots nonlinear fit object
    """
    fitted_columns = {
        col: col.replace("_", " ").capitalize() for col in fit.fcn.columns
    }
    plot_effective_beta = fit.fcn.beta_i_fcn is not None
    plot_hospitalizations = fit.fcn.sir_fcn.__name__ == "sihr_step"

    n_rows = 3 if plot_effective_beta or plot_hospitalizations else 2
    fig = make_subplots(
        rows=n_rows,
        cols=len(fitted_columns.keys()),
        subplot_titles=list(fitted_columns.values())
        + [col + " residual" for col in fitted_columns.values()]
        + ([r"Social distance [%]"] if plot_effective_beta else [])
        + ([r"Total admissions"] if plot_hospitalizations else []),
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )

    fcn = deepcopy(fit.fcn)
    fcn.as_array = False
    fcn.columns = None

    fit_res = fcn(fit.x, fit.p).rename(columns=fitted_columns)

    x = x if x is not None else fit_res.index

    for n_col, (col, yy) in enumerate(zip(fitted_columns.values(), fit.y.T)):
        n_col += 1
        add_gvar_scatter(
            fig,
            x=x,
            y=yy,
            name="Data",
            mode="markers+lines",
            color="#1f77b4",
            row=1,
            col=n_col,
            showlegend=n_col == 1,
        )
        add_gvar_scatter(
            fig,
            x=x,
            y=fit_res[col].values,
            name="Fit",
            mode="markers+lines",
            gv_mode="band",
            color="#bcbd22",
            row=1,
            col=n_col,
            showlegend=n_col == 1,
        )
        add_gvar_scatter(
            fig,
            x=x,
            y=yy,
            name="Data",
            mode="markers+lines",
            color="#1f77b4",
            row=1,
            col=n_col,
            showlegend=False,
        )
        add_gvar_scatter(
            fig,
            x=x,
            y=yy - fit_res[col].values,
            name="Residual",
            mode="markers+lines",
            color="#2ca02c",
            row=2,
            col=n_col,
            showlegend=False,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[0] * fit_res.shape[0],
                mode="lines",
                line_color="black",
                showlegend=n_col == 0,
            ),
            row=2,
            col=n_col,
        )

    i_col = 0
    if plot_effective_beta:
        i_col += 1
        add_gvar_scatter(
            fig,
            x=x,
            y=fit.fcn.beta_i_fcn(
                arange(fit.x["n_iter"]), **fit.fcn.convert_kwargs(fit.x, fit.p)
            )
            * 100,
            mode="markers+lines",
            gv_mode="band",
            color="#bcbd22",
            showlegend=False,
            col=i_col,
            row=3,
            y_max=100,
        )

    if plot_hospitalizations:
        i_col += 1
        capacity = fit.x["capacity"]
        add_gvar_scatter(
            fig,
            x=x,
            y=fit_res.hospitalized.values,
            gv_mode="band",
            row=3,
            col=i_col,
            color="#bcbd22",
            showlegend=False,
        )
        if any(
            capacity * 0.9
            < mean(fit_res.hospitalized.values) + sdev(fit_res.hospitalized.values)
        ):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=[capacity] * fit_res.shape[0],
                    mode="lines",
                    line_color="black",
                    name="Hospital capacity",
                ),
                row=3,
                col=2,
            )

    fig.update_layout(width=800, height=400 * n_rows)
    if plot_effective_beta:
        fig.update_layout(yaxis5=dict(range=[0, 100]))

    return fig


def plot_fit_range(  # pylint: disable=R0914
    fits: Dict[int, nonlinear_fit],
    col_wrap: int = 4,
    x: Optional[List[date]] = None,
    i_column: int = 1,
    y_max: Optional[float] = None,
):
    """Plots a range of fits
    """

    cols = col_wrap
    rows = len(fits) // col_wrap
    rows += 1 if cols * rows < len(fits) else 0

    nt_max = max(fits.keys())
    max_range_fit = fits[nt_max]

    yy = max_range_fit.y.T[i_column]
    xx = max_range_fit.x

    x_plot = x if x is not None else arange(xx["n_iter"]) * xx["bin_size"]

    fig = make_subplots(
        cols=cols,
        rows=rows,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=["Fitted days: {0}".format(nt * xx["bin_size"]) for nt in fits],
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )

    for i, (nt, fit) in enumerate(fits.items()):
        add_gvar_scatter(
            fig,
            x=x_plot,
            y=yy,
            gv_mode="scatter",
            mode="markers+lines",
            line_color="#1f77b4",
            col=i % col_wrap + 1,
            row=(i - i % col_wrap) // col_wrap + 1,
            name="Data",
            showlegend=i == 0,
        )
        add_gvar_scatter(
            fig,
            x=x_plot,
            y=fit.fcn(xx, fit.p).T[i_column],
            gv_mode="band",
            color="#bcbd22",
            col=i % col_wrap + 1,
            row=(i - i % col_wrap) // col_wrap + 1,
            name="Fit",
            showlegend=i == 0,
        )
        fig.add_trace(
            go.Scatter(
                x=x_plot[[nt - 2, nt - 2]],
                y=[0, y_max],
                mode="lines",
                line_color="black",
                name="Fit boundary",
                showlegend=i == 0,
            ),
            col=i % col_wrap + 1,
            row=(i - i % col_wrap) // col_wrap + 1,
        )

    fig.update_layout(
        yaxis_range=(0, y_max),
        **{f"yaxis{n+1}_range": (0, y_max) for n in range(1, len(fits))},
        width=800,
        height=300 * rows,
    )

    return fig
