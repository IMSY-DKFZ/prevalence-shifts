"""
General plotting utils for the prevalence-shifts project.
"""

import enum
import os
from pathlib import Path
from typing import Tuple, Optional, Sequence, Dict, Any, Union, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import ImageColor
from plotly import subplots

from src.prev.calibration import CalibrationMethod
from src.prev.metrics import Metric
from src.prev.quantification import QuantificationMethod

COLORS = ['#e6194b', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#ffe119', '#fabebe', '#f032e6',
          '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#ffd8b1', '#000075', '#808080',
          '#ffffff', '#bcf60c', '#000000']
COLORS = [ImageColor.getcolor(color, "RGB") for color in COLORS]
GREEN_1 = ImageColor.getcolor('#3cb44b', "RGB")  # ImageColor.getcolor("#A39F5D", "RGB")
GREEN_2 = ImageColor.getcolor('#808000', "RGB")
_pallete = COLORS


def _distribute_colors(enum_class, pivot_1=None, pivot_2=None) -> Dict[Any, Tuple[float, float, float, float]]:
    """
    Generates a map of elements to colors, based on an enum class and up to two pivot class instance.

    :param enum_class: enum class to distribute colors upon
    :param pivot_1: pivot element of enum class to receive a green color
    :param pivot_2: 2nd pivot element of enum class to receive a green color
    :return: a dictionary mapping elements of the enum class to colors
    """
    non_pivots = [elem for elem in enum_class if elem not in [pivot_1, pivot_2]]
    _map = {elem: _pallete[idx] for idx, elem in enumerate(non_pivots)}
    if pivot_1 is not None:
        _map[pivot_1] = GREEN_1
    if pivot_2 is not None:
        _map[pivot_2] = GREEN_2
    return _map


def get_metric_color(metric: Metric) -> Tuple[float, float, float, float]:
    """
    Gives consistent colors to identical metrics.

    :param metric: metric
    :return: color code as (red, green, blue, alpha)
    """
    return _distribute_colors(enum_class=Metric, pivot_1=Metric.EC_ADJUSTED, pivot_2=Metric.EC_EST)[metric]


def get_calibration_color(method: CalibrationMethod) -> Tuple[float, float, float, float]:
    """
    Gives consistent colors to identical calibration methods.

    :param method: calibration method
    :return: color code as (red, green, blue, alpha)
    """
    return _distribute_colors(enum_class=CalibrationMethod, pivot_1=CalibrationMethod.AFFINE_REWEIGHTED,
                              pivot_2=CalibrationMethod.AFFINE_ACC)[method]


def get_quantification_color(method: QuantificationMethod) -> Tuple[float, float, float, float]:
    """
    Gives consistent colors to identical calibration methods.

    :param method: calibration method
    :return: color code as (red, green, blue, alpha)
    """
    return _distribute_colors(enum_class=QuantificationMethod, pivot_1=QuantificationMethod.KDEyHD,
                              pivot_2=QuantificationMethod.KDEyML)[method]


class Confidence(enum.Enum):
    """How to display confidence intervals."""
    STD = "std"
    PERCENTILE = "percentile"
    NONE = "none"


def plot_aggregate_results(info_df: pd.DataFrame,
                           line_ids: Sequence[Union[str, Metric, CalibrationMethod, QuantificationMethod]] = tuple(),
                           metrics: Optional[Sequence[Metric]] = None,
                           file: Optional[Path] = None,
                           line_values_extraction_func: Optional[Callable] = None,
                           delta: bool = False,
                           bound: Optional[list] = None,
                           ci: Confidence = Confidence.STD,
                           y_axis_title: str = "Difference in metric's value",
                           title: Optional[str] = None,
                           font_size: float = 20.0,
                           opacity: float = 0.15,
                           size: Optional[Tuple[int, int]] = None,
                           line_width: float = 4.):
    """A shared plotting function for consistent figures. Generates a line plot."""
    # backwards compatibility (line_ids is the new "metrics")
    if len(line_ids) == 0 and metrics is None:
        metrics = (Metric.ACCURACY, Metric.F1, Metric.EC_ADJUSTED, Metric.MCC)
    if metrics is not None:
        if len(line_ids) != 0:
            raise ValueError('used both metrics and line_ids kwargs, metrics is only for backwards compatibility, '
                             'use line_ids only')
        line_ids = metrics
    # backwards compatibility (file default is now None and should be a Path)
    if not isinstance(file, Path):
        if file is not None:
            raise ValueError('better provide a full path instead of a string or something similar')

    x = list(np.arange(1, 10.5, 0.5))
    x_rev = x[::-1]

    def make_lines(results, line_name, color):
        y = list(np.nanmean(results, axis=0))
        if ci == Confidence.STD:
            y_upper = list(np.nanmean(results, axis=0) + np.nanstd(results, axis=0))
            y_lower = list(np.nanmean(results, axis=0) - np.nanstd(results, axis=0))
        elif ci == Confidence.PERCENTILE:
            y_upper = list(np.nanpercentile(results, 95, axis=0))
            y_lower = list(np.nanpercentile(results, 5, axis=0))
        if ci != Confidence.NONE:
            y_lower = y_lower[::-1]
            fig.add_trace(go.Scatter(
                x=x + x_rev,
                y=y_upper + y_lower,
                fill='toself',
                showlegend=False,
                name=line_name,
                line_color='rgba(255,255,255,0)',
                fillcolor="rgba" + str(color)[:-1] + f", {opacity})"

            ))
        fig.add_trace(go.Scatter(
            x=x, y=y,
            name=line_name,
            line_color="rgb" + str(color),
            line_width=line_width
        ))

    # pallete = [ImageColor.getcolor(color, "RGB") for color in px.colors.qualitative.Plotly]
    fig = go.Figure()
    for line_idx, line_identifier in enumerate(line_ids):
        if isinstance(line_identifier, Metric):
            name = line_identifier.value
            color = get_metric_color(line_identifier)
        elif isinstance(line_identifier, CalibrationMethod):
            name = line_identifier.value
            color = get_calibration_color(line_identifier)
        elif isinstance(line_identifier, QuantificationMethod):
            name = line_identifier.value
            color = get_quantification_color(line_identifier)
        else:
            name = line_identifier
            color = _pallete[line_idx]

        if line_values_extraction_func is None:
            # default case as previous
            results = np.stack((info_df[name]).values)
            if delta:
                val_results = np.stack(info_df["reference " + name].values)
                results = np.abs(results - np.tile(val_results, (results.shape[1], 1)).T)
                # metric+="_difference"
            else:
                results = np.abs(results)
        else:
            # apply new extraction scheme
            results = line_values_extraction_func(results_df=info_df, line_identifier=line_identifier, delta=delta)
        make_lines(results, name, color)
    fig.update_traces(mode='lines')
    if bound is not None:
        fig.update_layout(yaxis_range=bound)
    fig.update_layout(xaxis_title="Imbalance ratio", yaxis_title=y_axis_title, title=title, font_size=font_size,
                      template='plotly', legend_itemsizing='constant')
    if size is not None:
        fig.update_layout(autosize=False, width=size[0], height=size[1])
    fig['layout']['font']['family'] = "NewComputerModern10"
    if file:
        fig.write_image(file.with_suffix(suffix='.svg'))
        fig.write_image(file.with_suffix(suffix='.png'))
        fig.write_html(file.with_suffix(suffix='.html'))
    return fig


def box_plot(df: pd.DataFrame,
             line_ids: Sequence[Union[str, Metric, CalibrationMethod]] = tuple(),
             metrics: Optional[Sequence[Metric]] = None,
             font_size: float = 30.):
    """A shared plotting function for consistent figures. Generates a box plot."""
    # backwards compatibility (line_ids is the new "metrics")
    if len(line_ids) == 0 and metrics is None:
        metrics = (Metric.ACCURACY, Metric.F1, Metric.EC_ADJUSTED, Metric.MCC)
    if metrics is not None:
        if len(line_ids) != 0:
            raise ValueError('used both metrics and line_ids kwargs, metrics is only for backwards compatibility, '
                             'use line_ids only')
        line_ids = metrics
    fig = go.Figure()
    for line_idx, line_identifier in enumerate(line_ids):
        if isinstance(line_identifier, Metric):
            name = line_identifier.value
            color = get_metric_color(line_identifier)
        elif isinstance(line_identifier, CalibrationMethod):
            name = line_identifier.value
            color = get_calibration_color(line_identifier)
        elif isinstance(line_identifier, QuantificationMethod):
            name = line_identifier.value
            color = get_quantification_color(line_identifier)
        else:
            name = line_identifier
            color = _pallete[line_idx]
        fig.add_trace(go.Box(
            y=df[name],
            name=name,
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            marker_color="rgb" + str(color),  # add the box plot color
            marker_opacity=0.6,
            pointpos=0  # move points to middle of the box
        ))
    fig.update_layout(legend={"font": {"size": font_size}}, font_size=font_size)
    return fig


def multiplot(rows: int = 2, cols: int = 2, subplts: Sequence = tuple(), x_title: Optional[str] = None,
              y_title: Optional[str] = None, row_titles: Optional[Sequence] = None,
              column_titles: Optional[Sequence] = None,
              shared_xaxes: bool = True, shared_yaxes: bool = True, legend_index: int = 0,
              bold_titles: bool = True, horizontal_spacing: float = 0.02, vertical_spacing: float = 0.03,
              sub_x_axis_titles: Optional[Dict] = None, sub_y_axis_titles: Optional[Dict] = None,
              sub_x_ranges: Optional[Dict] = None, sub_y_ranges: Optional[Dict] = None, width: int = 1800,
              height: int = 500, ir_axes: Optional[list] = None, ir_values: Optional[list] = None,
              little_guys: Optional[bool] = False, icon_axes: Optional[list] = [1], icon_size: Optional[float] = 0.1,
              icon_y_adjustment: Optional[float] = 0.04):
    """
    Plots multiple plots in a single figure.

    :param rows: number of rows in the multiplot
    :param cols: number of columns in the multiplot
    :param subplts: sequence of subplots
    :param x_title: x axis title
    :param y_title: y axis title
    :param row_titles: sequence of row titles
    :param column_titles: sequence of column titles
    :param shared_xaxes: 
    :param shared_yaxes:
    :param legend_index: index of subplot after which the legend is inherited
    :param bold_titles:
    :param horizontal_spacing: 
    :param vertical_spacing:
    :param sub_x_axis_titles: dictionary of index-title key-value pairs for the x axes
    :param sub_y_axis_titles: dictionary of index-title key-value pairs for the y axes
    :param sub_x_ranges: dictionary of index-range key-value pairs for the x axes 
    :param sub_y_ranges: dictionary of index-range key-value pairs for the y axes
    :param: width: width of the figure
    :param height: height of the figure
    :param ir: add a box with ir value to 2nd figure
    :return fig: 
    """
    if len(subplts) != rows * cols:
        raise RuntimeError("Given a different number of subplots than the size of the grid (rows times cols)")
    if bold_titles:
        if x_title:
            x_title = "<b>" + x_title + "</b>"
        if y_title:
            y_title = "<b>" + y_title + "</b>"
        if row_titles:
            row_titles = ["<b>" + t + "</b>" for t in row_titles]
        if column_titles:
            column_titles = ["<b>" + t + "</b>" for t in column_titles]
        if sub_x_axis_titles:
            sub_x_axis_titles = {k: "<b>" + sub_x_axis_titles[k] + "</b>" for k in sub_x_axis_titles.keys()}
        if sub_y_axis_titles:
            sub_y_axis_titles = {k: "<b>" + sub_y_axis_titles[k] + "</b>" for k in sub_y_axis_titles.keys()}
    fig = subplots.make_subplots(rows=rows, cols=cols,
                                 y_title=y_title,
                                 x_title=x_title,
                                 row_titles=row_titles,
                                 column_titles=column_titles,
                                 horizontal_spacing=horizontal_spacing,
                                 vertical_spacing=vertical_spacing,
                                 shared_xaxes=shared_xaxes,
                                 shared_yaxes=shared_yaxes)
    fig.update_layout(template='plotly')
    fig.update_annotations(font_size=20, font_color='black', bgcolor='white')
    for i, subplt in enumerate(subplts):
        for value in subplt['data']:
            if i != legend_index:
                value['showlegend'] = False
            fig.add_trace(value, row=int(i / cols) + 1, col=i % cols + 1)

    if sub_x_axis_titles:
        for k in sub_x_axis_titles.keys():
            subplt_count = "" if k == 0 else str(k + 1)
            fig['layout']['xaxis' + subplt_count]['title'] = sub_x_axis_titles[k]
    if sub_y_axis_titles:
        for k in sub_y_axis_titles.keys():
            subplt_count = "" if k == 0 else str(k + 1)
            fig['layout']['yaxis' + subplt_count]['title'] = sub_y_axis_titles[k]
    if sub_x_ranges:
        for k in sub_x_ranges.keys():
            subplt_count = "" if k == 0 else str(k + 1)
            fig['layout']['xaxis' + subplt_count]['range'] = sub_x_ranges[k]
    if sub_y_ranges:
        for k in sub_y_ranges.keys():
            subplt_count = "" if k == 0 else str(k + 1)
            fig['layout']['yaxis' + subplt_count]['range'] = sub_y_ranges[k]

    fig['layout']['font']['size'] = 22
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=30)
    fig['layout']['font']['family'] = "NewComputerModern10"
    fig.update_layout(autosize=False, width=width, height=height,
                      legend={"font": {"size": 30}, "itemsizing": 'constant'}, margin=dict(t=10, b=90))
    if little_guys:
        for axis_no in icon_axes:
            add_little_guys(fig, axis_no, size=icon_size, y_adjustment=icon_y_adjustment)
    if ir_values is not None:
        for axis, ir in zip(ir_axes, ir_values):
            add_ir_annotation(fig, axis, ir)
    return fig


def add_little_guys(fig, subplot_axis_no, size=0.1, y_adjustment=0.04):
    """Adds two figures demonstrating the imbalance ration with colored little guys."""
    if subplot_axis_no == 1:
        xname = "xaxis"
        yname = 'yaxis'
    else:
        xname = f'xaxis{subplot_axis_no}'
        yname = f'yaxis{subplot_axis_no}'
    x_balanced, x_high_ir = fig['layout'][xname]['domain']
    y_pos, y_top = fig['layout'][yname]['domain']
    y_pos -= y_adjustment
    fig.update_layout(
        images=fig.layout.images + (
            go.layout.Image(
                source=str(Path(os.getcwd()).parent / "data/little_guys_balanced.png"),
                # URL or local path to the image
                xref="paper", yref="paper",
                x=x_balanced, y=y_pos,
                sizex=size, sizey=size,  # Size of the image
                xanchor="left", yanchor="top"
            ),
            go.layout.Image(
                source=str(Path(os.getcwd()).parent / "data/little_guys_high_ir.png"),  # URL or local path to the image
                xref="paper", yref="paper",
                x=x_high_ir, y=y_pos,
                sizex=size, sizey=size,  # Size of the image
                xanchor="right", yanchor="top"
            )
        )

    )


def add_ir_annotation(fig, subplot_axis_no, ir_value=10):
    """Adds a small box displaying the current imbalance ratio to an existing figure."""
    suffix = str(subplot_axis_no) if subplot_axis_no != 1 else ""
    fig.add_annotation(
        text=f"IR={ir_value}",
        x=fig['layout'][f'xaxis{suffix}']['domain'][1],
        y=fig['layout'][f'yaxis{suffix}']['domain'][1],
        xref='paper', yref='paper',  # Reference to subplot
        xanchor='right', yanchor='top',  # Anchor point of text (right-top corner)
        showarrow=False,  # No arrow for annotation
        font=dict(size=22, color="black"),  # Font settings
        bgcolor="white",  # Background color of the annotation box
        bordercolor="black",  # Border color of the annotation box
        borderwidth=1,  # Border width of the annotation box
    )

