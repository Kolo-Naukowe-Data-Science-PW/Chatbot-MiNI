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
    if type(column_names) is not list:
        return Exception("Incorrect columns list, is not a python list")

    if not os.path.isfile(file_path):
        return Exception("File path is not a file")

    df = pd.read_csv(file_path)
    df = df[column_names]
    return df


def convert_data_frame_to_string(data_frame: pd.DataFrame):
    """
    The function `convert_data_frame_to_string` converts a pandas DataFrame to a NumPy array of strings.

    :param data_frame: A pandas DataFrame containing two columns
    :type data_frame: pd.DataFrame
    """

    if len(data_frame.columns) != 2:
        return Exception("Wrong number of columns")

    return data_frame.to_numpy(dtype="str")


def get_data(file_path, column_names):
    """
    The function `get_data` reads data from a CSV file and converts it to a string format.

    :param file_path: The `file_path` parameter is a string that represents the path to the CSV file
    from which data will be read
    :param column_names: The `column_names` parameter is a list of column names that you want to extract
    from the CSV file located at the `file_path`. These column names will be used to read specific
    columns from the CSV file and convert the data frame into a string format
    :return: The function `get_data` is returning the data from a CSV file located at `file_path` with
    the specified `column_names` as a string after converting it to a data frame.
    """
    return convert_data_frame_to_string(read_data_from_csv(file_path, column_names))
