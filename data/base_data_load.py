import pandas as pd

def load_census(filename='data/census.xlsx', keep_default_na=False,
                filters=[(lambda x: x['any_padro'] == 1940)]):
    WS = pd.read_excel(filename, keep_default_na=keep_default_na)
    for f in filters:
        WS = WS[WS.apply(f, axis=1)]
    return WS