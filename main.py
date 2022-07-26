from dateutil import relativedelta
import numpy as np
import pandas as pd

from dash import Dash, html, dcc
import plotly.graph_objects as go

from data_extraction import get_ipca_data, get_bvsp_data

def make_return_matrix(time_series: pd.Series, freq: str) -> pd.DataFrame:
    frequency_data = time_series.groupby(pd.Grouper(freq=freq)).last()
    idx_values = frequency_data.index.values

    # TODO: Anualizar retornos
    return_func = lambda date: (frequency_data.values / frequency_data[date]) - 1
    return_matrix_values = np.array([return_func(date) for date in idx_values])
    mask = ~np.triu(np.ones_like(return_matrix_values, dtype=bool), k=1)
    
    return_matrix = pd.DataFrame(return_matrix_values, index=idx_values, columns=idx_values).mask(mask)
    return_matrix = return_matrix.iloc[:-1, 1:] # Remove primeira coluna e última linha
    return return_matrix

def main():
    freq = "Y"

    ticker_data = get_bvsp_data()
    bvsp_return_matrix = make_return_matrix(ticker_data, freq)
    
    ipca_data = get_ipca_data()
    ipca_return_matrix = make_return_matrix(ipca_data, freq)

    return_matrix = bvsp_return_matrix - ipca_return_matrix

    return_matrix = return_matrix * 100

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=return_matrix.values,
                y=return_matrix.index,
                x=return_matrix.columns,
                colorscale='Viridis',
                hoverongaps=False,
                hovertemplate='De %{y} até %{x}: <b>%{z:.2f}%<b><extra></extra>',
                showlegend=False,
                showscale=False
            )
        ]
    )
    fig.update_xaxes(
        fixedrange=True, side="top", tickmode='array', 
        tickvals=[return_matrix.columns[0], return_matrix.columns[-1]] + return_matrix.columns[return_matrix.columns.year % 5 == 0].to_list(),
        tickformat='%Y')
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
    app.layout = html.Div(
        className="parent", 
        children=[
            dcc.Graph(
                id='example-graph',
                figure=fig,
                className="center-me"
            )
    ])

    app.run_server(debug=True)

if __name__ == "__main__":
    main()