# from src.chat_bot_auto_evaluation.evaluate import *
# from src.chat_bot_auto_evaluation.metrics.metrics import *
import pandas as pd

from src.chat_bot_auto_evaluation.read_data import read_data_from_csv

FILE_PATH = "src/chat_bot_auto_evaluation/tests/test.csv"
COLUMNS = ["a", "b"]


def test_read_data():
    """this test function checks if output of a read_data_from_csv function is valid"""
    assert type(read_data_from_csv(FILE_PATH, COLUMNS)) is pd.DataFrame
    assert type(read_data_from_csv("no_file.csv", COLUMNS)) is Exception
    assert type(read_data_from_csv(FILE_PATH[:-3], COLUMNS)) is Exception
    assert type(read_data_from_csv(FILE_PATH, COLUMNS[0])) is Exception
