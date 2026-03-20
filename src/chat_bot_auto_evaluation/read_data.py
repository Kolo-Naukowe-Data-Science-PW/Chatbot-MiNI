import os

import pandas as pd


def read_data_from_csv(file_path, column_names):
    """_summary_

    Args:
        file_path (str): path to csv file with data
        column_names (list): list with all the column names you want to use

    Returns:
        pandas data frame: returns columns from file_path
    """

    # checking if file_path is a csv

    if file_path[-3:] != "csv":
        return Exception("Incorrect file path, path is not a csv file")

    # check if column names is a list
    if type(column_names) is list:
        return Exception("Incorrect columns list, is not a python list")

    if not os.path.isfile(file_path):
        return Exception("File path is not a file")

    df = pd.read_csv(file_path)
    df = df[column_names]
    return df
