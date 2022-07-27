from datetime import date
import os

import pandas as pd
import requests
import yfinance as yf

INITIAL_DATE = date(1995, 1, 1)

CACHED_DATA_DIR = "cached_data"
if not os.path.exists(CACHED_DATA_DIR):
    os.makedirs(CACHED_DATA_DIR)

def get_ipca_data():
    # TODO: improve exception handling
    try:
        data = get_ibge_ipca_data()
    except:
        return get_cached_ipca_data()
    
    try:
        cache_ipca_data(data)
    except:
        pass

    return data

def get_ibge_ipca_data() -> pd.Series:
    year_months = [f"{dt.year}{dt.month:02}" for dt in pd.date_range(INITIAL_DATE, date.today(), freq="M")]
    url = f"https://servicodados.ibge.gov.br/api/v3/agregados/1737/periodos/{'|'.join(year_months)}/variaveis/2266?localidades=N1[all]"

    with requests.Session() as s:
        response = s.get(url)

    data = response.json()
    data_serie_temporal = data[0]['resultados'][0]['series'][0]['serie']

    serie_ipca = pd.DataFrame.from_dict(data_serie_temporal, orient="index", columns=["ipca"], dtype=float).squeeze()
    serie_ipca.index = pd.to_datetime(serie_ipca.index, format="%Y%m")
    serie_ipca.index.name = "date"

    return serie_ipca

def cache_ipca_data(ipca_data: pd.Series) -> None:
    filepath = os.path.join(CACHED_DATA_DIR, "ipca.csv")
    ipca_data.to_csv(filepath)

def get_cached_ipca_data() -> pd.Series:
    return pd.read_csv("data\\ipca.csv", index_col=0, parse_dates=True).squeeze()

def get_bvsp_data() -> pd.Series:
    # TODO: improve exception handling
    try:
        data = get_bvsp_yf_data()
    except:
        return get_cached_bvsp_data()
    
    try:
        cache_bvsp_data(data)
    except:
        pass

    return data

def get_bvsp_yf_data() -> pd.Series:
    ticker_name = "^BVSP"
    ticker = yf.Ticker(ticker_name)
    data = ticker.history(period="max", interval="1d", start=INITIAL_DATE)
    return data.Close

def get_cached_bvsp_data() -> pd.Series:
    return pd.read_csv("data\\bvsp.csv", index_col=0, parse_dates=True).squeeze()

def cache_bvsp_data(bvsp_data: pd.Series) -> None:
    filepath = os.path.join(CACHED_DATA_DIR, "bvsp.csv")
    bvsp_data.to_csv(filepath)