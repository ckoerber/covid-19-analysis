"""Functions to simplify plotting
"""

import plotly.graph_objects as go
import gvar as gv


def hex_to_rgba(hex_str, alpha):
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
        kwargs.pop("mode")
        name = kwargs.pop("name")
        showlegend = kwargs.pop("showlegend", True)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[max(yy, 0) for yy in (y_mean - y_sdev)],
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
