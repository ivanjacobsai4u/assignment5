import pytest
import pandas as pd
import os

df_train = pd.read_csv("data/train.csv", sep=',')
df_test = pd.read_csv("data/test.csv")

def test_train_data():

        assert df_train.isnull().sum().any == False

def test_test_data():

        assert df_test.isnull().sum().any() == False