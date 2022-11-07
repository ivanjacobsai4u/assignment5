import pytest
import pandas as pd
import os

df_train = pd.read_csv("data/train.csv", sep=',')
df_test = pd.read_csv("data/test.csv")


def test_train_data():
    assert df_train.isnull().sum().any == False


def test_test_data():
    assert df_test.isnull().sum().any() == False


def test_outliers_numerical_columns():
    # Calculate outliers for all numerical values
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    df_numerics = df_train.select_dtypes(include=numerics)
    Q1 = df_numerics.quantile(0.25)
    Q3 = df_numerics.quantile(0.75)
    OF = 3.0
    EVF = 6.0
    IQR = Q3 - Q1
    outliers = (
        (
                (((Q1 - EVF * IQR) <= df_numerics) & (df_numerics < (Q1 - OF * IQR)))
                | ((df_numerics <= (Q3 + EVF * IQR)) & (df_numerics > (Q3 + OF * IQR)))
        )
        .transpose()
        .any()
        .sum()
    )

    assert outliers == 0


def test_extreme_values_numerical_columns():
    # Calculate outliers for all numerical values
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    df_numerics = df_train.select_dtypes(include=numerics)
    Q1 = df_numerics.quantile(0.25)
    Q3 = df_numerics.quantile(0.75)
    OF = 3.0
    EVF = 6.0
    IQR = Q3 - Q1

    extreme_values = len(
        df_numerics[
            ((df_numerics < (Q1 - EVF * IQR)) | (df_numerics > (Q3 + EVF * IQR)))
            .transpose()
            .any()
        ]
    )

    assert extreme_values == 0
