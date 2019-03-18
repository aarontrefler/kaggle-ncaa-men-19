"""Utility support"""
import pandas as pd

def cols_to_front(df, front_cols):
    """Moves selected coumns to front of DataFrame"""
    cols = list(df)
    front_cols.reverse()
    for col in front_cols:
        cols.insert(0, cols.pop(cols.index(col)))

    return df.loc[:, cols]


def display_df(df, n=1, tail=False, title=None):
    """Custom display method for DataFrames"""
    if title:
        print(title + ':')
    display(df.head(n), df.tail(n), df.shape) if tail else display(df.head(n), df.shape)
