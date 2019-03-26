"""Utility support"""
import datetime

import pandas as pd

project_path = "/Users/aarontrefler_temp2/Documents/My Documents/Kaggle/kaggle-ncaa-men-19/"

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


def create_datestamp():
    """Return current datestamp formatted as YYYYMMDDHH"""
    now = datetime.datetime.now()
    return "{year}{month:02}{day:02}{hour:02}".format(
        year=now.year, month=now.month, day=now.day, hour=now.hour)
