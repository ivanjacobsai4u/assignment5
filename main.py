import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df_train = pd.read_csv("data/train.csv", sep=',')
df_test = pd.read_csv("data/test.csv")


# Replace NaN's with zero
df_train.replace(np.nan, 0,inplace=True)
df_test.replace(np.nan, 0,inplace=True)

#Drop the highly correlated columns from the dataframe
x_train = df_train.loc[:, df_train.columns != "SalePrice"]
x_corr_matrix = x_train.corr().abs()

upper_triangle = x_corr_matrix.where(np.triu(np.ones(x_corr_matrix.shape), k=1).astype(np.bool))
column_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.70)]

df_train = df_train.drop(columns= column_drop);
df_train.reset_index(drop=True, inplace=True)

# Remove outliers
numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
df_numerics = df_train.select_dtypes(include=numerics)
Q1 = df_numerics.quantile(0.25)
Q3 = df_numerics.quantile(0.75)
OF = 3.0
EVF = 6.0
IQR = Q3 - Q1
outliers =  (
                (((Q1 - EVF * IQR) <= df_numerics) & (df_numerics < (Q1 - OF * IQR)))
                | ((df_numerics <= (Q3 + EVF * IQR)) & (df_numerics > (Q3 + OF * IQR)))
        ).transpose().any()


extreme_values=((df_numerics < (Q1 - EVF * IQR)) | (df_numerics > (Q3 + EVF * IQR))).transpose().any()
df_train=df_train.loc[~outliers]
df_train=df_train.loc[~extreme_values]
df_numerics = df_train.select_dtypes(include=numerics)




df_train = df_train.select_dtypes(include=numerics)

def test_train_data():
    assert df_train.isna().sum().any() == False


def test_test_data():
    assert df_test.isna().sum().any() == False


def test_outliers_numerical_columns():
    # Calculate outliers for all numerical values

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


    extreme_values = len(
        df_numerics[
            ((df_numerics < (Q1 - EVF * IQR)) | (df_numerics > (Q3 + EVF * IQR)))
            .transpose()
            .any()
        ]
    )

    assert extreme_values == 0

def test_features_high_correlation():
    x_train = df_train.loc[:, df_train.columns != "SalePrice"]
    x_corr_matrix = x_train.corr().abs()
    upper_triangle = x_corr_matrix.where(np.triu(np.ones(x_corr_matrix.shape), k=1).astype(np.bool))
    assert len( [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.70)])==0



SalePrices=df_train.pop('SalePrice')
X_train, X_test, y_train, y_test =train_test_split(df_train, SalePrices, test_size=0.2, random_state=1)

regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X_train, y_train)

print("R^2 coefficient of determination of the prediction: ",regr.score(X_test,y_test))
