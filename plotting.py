"""Functions to simplify plotting
"""
from copy import deepcopy

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import gvar as gv
from lsqfit import nonlinear_fit
from gvar import fmt_values, fmt_errorbudget

from pandas import DataFrame

from models.utils import get_doubling_time


def hex_to_rgba(hex_str: str, alpha: float):
    """Converts hex string to rgba
    """
    if "#" in hex_str:
        hex_str = hex_str.replace("#", "")
    out = "rgba({0},{1},{2},{alpha})".format(
        *[int(hex_str[i : i + 2], 16) for i in (0, 2, 4)], alpha=alpha
    )
    return out


def add_gvar_scatter(fig: go.Figure, gv_mode: str = "scatter", **kwargs) -> go.Figure:
    """Wraps adds scatter for gvars
    """
    x = kwargs.pop("x")
    y = kwargs.pop("y")

    y_mean = gv.mean(y)  # pylint: disable=E1101
    y_sdev = gv.sdev(y)  # pylint: disable=E1101

    row, col = kwargs.pop("row", None), kwargs.pop("col", None)
    color = kwargs.pop("color", None)

    if gv_mode == "band":
        kwargs.pop("mode", None)
        name = kwargs.pop("name", None)
        showlegend = kwargs.pop("showlegend", True)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[max(yy, 0) for yy in y_mean - y_sdev],
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
                y=(y_mean + y_sdev),
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
        < gv.mean(df_sihr.hospitalized.values) + gv.sdev(df_sihr.hospitalized.values)
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
    "inital_doubling_time": "t2(0) [days]",
    "recovery_days": "Recovery [days]",
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


def plot_fits(fit: nonlinear_fit) -> go.Figure:
    """Plots nonlinear fit object
    """
    fitted_columns = [col.replace("_", " ").capitalize() for col in fit.fcn.columns]
    fig = make_subplots(
        rows=2,
        cols=len(fitted_columns),
        subplot_titles=fitted_columns + [col + " residual" for col in fitted_columns],
    )

    fcn = deepcopy(fit.fcn)
    fcn.as_array = False
    fcn.columns = None

    fit_res = fcn(fit.x, fit.p)

    for n_col, (col, yy) in enumerate(zip(fitted_columns, fit.y.T)):
        plotting.add_gvar_scatter(
            fig,
            x=fit_res.index,
            y=yy,
            name="Data",
            mode="markers+lines",
            color="#1f77b4",
            row=1,
            col=n_col,
            showlegend=n_col == 0,
        )
        plotting.add_gvar_scatter(
            fig,
            x=fit_res.index,
            y=fit_res[col].values,
            name="Fit",
            mode="markers+lines",
            gv_mode="band",
            color="#bcbd22",
            row=1,
            col=n_col,
            showlegend=n_col == 0,
        )
        plotting.add_gvar_scatter(
            fig,
            x=fit_res.index,
            y=yy,
            name="Data",
            mode="markers+lines",
            color="#1f77b4",
            row=1,
            col=n_col,
        )
        plotting.add_gvar_scatter(
            fig,
            x=fit_res.index,
            y=y - fit_res[col].values,
            name="Residual",
            mode="markers+lines",
            color="#2ca02c",
            row=2,
            col=n_col,
            showlegend=n_col == 0,
        )
        fig.add_trace(
            go.Line(x=fit_res.index, y=[0] * fit_res.shape[0], color="black"),
            row=2,
            col=n_col,
            showlegend=n_col == 0,
        )

    return fig
