import numpy as np
import pandas as pd

from dash import Dash, html, dcc
import plotly.graph_objects as go

from data_extraction import get_ipca_data, get_bvsp_data

def annualized_return(start_date: np.datetime64, start_val: np.float64, end_date: np.datetime64, end_val: np.float64) -> np.float64:
    if start_date == end_date:
        return 0
    
    start_year = start_date.astype('datetime64[Y]').astype(int) + 1970
    end_year = end_date.astype('datetime64[Y]').astype(int) + 1970

    return (end_val / start_val) ** (1 / (end_year - start_year)) - 1

def make_return_matrix(time_series: pd.Series, freq: str) -> pd.DataFrame:
    frequency_data = time_series.groupby(pd.Grouper(freq=freq)).last()
    idx_values = frequency_data.index.values

    ann_return_matrix_values = np.empty((len(idx_values), len(idx_values)))
    for i in range(len(idx_values)-1):
        for j in range(i+1, len(idx_values)):
            ann_return_matrix_values[i, j] = annualized_return(idx_values[i], frequency_data.values[i], idx_values[j], frequency_data.values[j])

    mask = ~np.triu(np.ones_like(ann_return_matrix_values, dtype=bool), k=1)
    
    return_matrix = pd.DataFrame(ann_return_matrix_values, index=idx_values, columns=idx_values).mask(mask)
    return_matrix = return_matrix.iloc[:-1, 1:] # Remove primeira coluna e última linha
    return return_matrix

def map_between_zero_and_one(value: float, min_value: float, max_value: float) -> float:
    return (value - min_value) / (max_value - min_value)

freq = "Y"

ticker_data = get_bvsp_data()
bvsp_return_matrix = make_return_matrix(ticker_data, freq)

ipca_data = get_ipca_data()
ipca_return_matrix = make_return_matrix(ipca_data, freq)

return_matrix = bvsp_return_matrix - ipca_return_matrix

return_matrix = return_matrix * 100
min_return = return_matrix.min().min()
max_return = return_matrix.max().max()

return_matrix_zero_to_one = (return_matrix - min_return) / (max_return - min_return)

fig = go.Figure(
    data=[
        go.Heatmap(
            z=return_matrix_zero_to_one.values,
            y=return_matrix_zero_to_one.index,
            x=return_matrix_zero_to_one.columns,
            text=return_matrix.values,
            hoverongaps=False,
            hovertemplate='De %{y} até %{x}: <b>%{text:.2f}%<b><extra></extra>',
            showlegend=False,
            showscale=False,
            colorscale=[
                [0, "rgb(159, 83, 75)"],
                [map_between_zero_and_one(0, min_return, max_return), "rgb(159, 83, 75)"], 
                [map_between_zero_and_one(0, min_return, max_return), "rgb(198, 158, 144)"], 
                [map_between_zero_and_one(3, min_return, max_return), "rgb(198, 158, 144)"], 
                [map_between_zero_and_one(3, min_return, max_return), "rgb(204, 204, 181)"], 
                [map_between_zero_and_one(7, min_return, max_return), "rgb(204, 204, 181)"], 
                [map_between_zero_and_one(7, min_return, max_return), "rgb(155, 166, 118)"], 
                [map_between_zero_and_one(10, min_return, max_return), "rgb(155, 166, 118)"], 
                [map_between_zero_and_one(10, min_return, max_return), "rgb(122, 135, 75)"], 
                [1, "rgb(122, 135, 75)"]
            ],
        )
    ]
)
fig.update_xaxes(
    fixedrange=True, side="top", tickmode='array', 
    tickvals=[return_matrix.columns[0], return_matrix.columns[-1]] + return_matrix.columns[return_matrix.columns.year % 5 == 0].to_list(),
    tickformat='%Y', tickangle=-60)
fig.update_yaxes(autorange="reversed", fixedrange=True, showticklabels=False)
fig.update_layout(title='Return Matrix', plot_bgcolor='#FFFFFF')

fig.update_layout(
    annotations=[
        dict(
            x=xi,
            y=yi,
            text=yi.strftime('%Y'),
            showarrow=False,
            borderwidth=2,
            xanchor="right",
        ) 
    for xi, yi in zip(return_matrix.columns[:-1], return_matrix.index[1:]) if ((yi.year % 5 == 0) or yi in (return_matrix.index[0], return_matrix.index[-1]))] \
    + [
        dict(
            x=return_matrix.index[0],
            y=return_matrix.index[0],
            text=return_matrix.index[0].strftime('%Y'),
            showarrow=False,
            borderwidth=2,
            xanchor="right",
        )
    ]
)

app = Dash(__name__)

server = app.server

app.layout = html.Div(
    className="parent", 
    children=[
        dcc.Graph(
            id='example-graph',
            figure=fig,
            className="center-me"
        )
])


if __name__ == "__main__":
    app.run_server(debug=True)