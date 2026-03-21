import numpy as np
import pandas as pd
import torch

from chat_bot_auto_evaluation.metrics import Metrics
from chat_bot_auto_evaluation.prepare_data import (
    convert_data_frame_to_string,
    get_data,
    read_data_from_csv,
)

FILE_PATH = "src/chat_bot_auto_evaluation/tests/test.csv"
COLUMNS = ["a", "b"]


def test_read_data():
    """this test function checks if output of a read_data_from_csv function is valid"""
    assert type(read_data_from_csv(FILE_PATH, COLUMNS)) is pd.DataFrame
    assert type(read_data_from_csv("no_file.csv", COLUMNS)) is Exception
    assert type(read_data_from_csv(FILE_PATH[:-3], COLUMNS)) is Exception
    assert type(read_data_from_csv(FILE_PATH, COLUMNS[0])) is Exception


def test_convert_data_frame_to_string():
    df = read_data_from_csv(FILE_PATH, COLUMNS)
    assert type(convert_data_frame_to_string(df)) is np.ndarray
    assert len(convert_data_frame_to_string(df)) == 2


def test_get_data():
    assert type(get_data(FILE_PATH, COLUMNS)) is np.ndarray
    assert len(get_data(FILE_PATH, COLUMNS)) == 2


# testing metrics class and methods

candidates = ["Pogoda jest dzisiaj naprawdę piękna."]
references = ["Dzisiaj na zewnątrz jest cudowny dzień."]


class TestMetrics:

    def test_init(self):
        object = Metrics(candidates, references)
        assert object.candidates == candidates
        assert object.references == references

    def test_bert_score(self):
        object = Metrics(candidates, references)
        P, R, F1 = object.bert_score()
        assert type(P) is torch.Tensor
        assert type(R) is torch.Tensor
        assert type(F1) is torch.Tensor
