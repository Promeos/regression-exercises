import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression, RFE


def prep_data(df, modeling=False):
    '''
    Signature: prep_data(df, modeling=False)
    Docstring:

    Parameters
    ----------

    Returns
    -------

    '''


def add_encoded_columns(df, drop_encoders=True):
    '''
    Signature: add_encoded_columns(df, drop_encoders=True)
    Docstring:

    Parameters
    ----------

    Returns
    -------

    '''
    columns_to_encode = df.select_dtypes('O').columns.to_list()
    encoded_columns = pd.get_dummies(df[columns_to_encode], drop_first=True, dummy_na=False)

    df = pd.concat([df, encoded_columns], axis=1)
    
    if drop_encoders:
        df =  df.drop(columns=columns_to_encode)
        return df
    else:
        return df, encoded_columns


def split_data(df):
    '''
    Signature: split_data(df)
    Docstring:

    Parameters
    ----------

    Returns
    -------

    '''
    train_validate, test = train_test_split(df, test_size=.20, random_state=369)
    train, validate = train_test_split(train_validate, test_size=.20, random_state=369)
    return train, validate, test


def attributes_target_split(df, target):
    '''
    Signature: attributes_target_split(df, target)
    Docstring:

    Parameters
    ----------

    Returns
    -------

    '''
    x = df.drop(columns=target)
    y = df[target]
    return x, y


def add_scaled_columns(train, validate, test, scaler):
    '''
    Signature: add_scaled_columns(train, validate, test, scaler)
    Docstring:

    Parameters
    ----------

    Returns
    -------

    '''
    columns_to_scale = train.columns.to_list()
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    scaler.fit(train[columns_to_scale])

    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]),
                     columns=new_column_names,
                     index=train.index
                     )],axis=1)
    
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                     columns=new_column_names,
                     index=validate.index
                     )], axis=1)
    
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]),
                     columns=new_column_names,
                     index=test.index
                     )], axis=1)
    return train, validate, test


def features_for_modeling(predictors, target, k_features):
    '''
    Signature: features_for_modeling(predictors, target, k_features)
    Docstring:

    Parameters
    ----------

    Returns
    -------

    '''
    df_best = pd.DataFrame(select_kbest(predictors, target, k_features))
    df_rfe = pd.DataFrame(rfe(predictors, target, k_features))
    
    df_features = pd.concat([df_best, df_rfe], axis=1)
    return df_features


def select_kbest(predictors, target, k_features=3):
    '''
    Signature: select_kbest(predictors, target, k_features=3)
    Docstring:

    Parameters
    ----------

    Returns
    -------

    '''
    f_selector = SelectKBest(f_regression, k=k_features)
    f_selector.fit(predictors, target)
    
    f_mask = f_selector.get_support()
    f_features = predictors.iloc[:,f_mask].columns.to_list()
    
    print(f"Select K Best returned {len(f_features)} features")
    print(f_features)
    return predictors[f_features]
    
    
def rfe(predictors, target, k_features=3):
    '''
    Signature: rfe(predictors, target, k_features=3)
    Docstring:

    Parameters
    ----------

    Returns
    -------

    '''
    lm = LinearRegression()
    rfe = RFE(lm, k_features)

    rfe_mask = rfe.support_    
    rfe_features = predictors.iloc[:, rfe_mask].columns.to_list()

    print(f"Recursive Feature Elimination: {len(rfe_features)} features")
    print(rfe_features)
    return(predictors[rfe_features])