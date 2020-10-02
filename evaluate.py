import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt

from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error


def plot_residuals(actual, predicted):
    '''
    Signature: plot_residuals(actual, predicted)
    Docstring:
    Plots the `actual` outcome v. the residual of `predicted` values

    Parameters
    ----------
    actual : pandas Series
        The actual outcome. Stored in columns with names like:
        y, target
        It's the column you're trying to predict.
    predicted : pandas Series
        The predicted outcome. Stored in columns with names like:
        yhat, predicted
        It's the column that we use to predict the `actual` column.
        
    Returns
    -------
    A plot of actual values v. residuals of predicted.
    '''
    residuals = actual - predicted
    plt.hlines(0, actual.min(), actual.max(), ls=':')
    plt.scatter(actual, residuals)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Residual')
    return plt.gca()


def ols_model_evaluation(data, predictor='', target=''):
    '''
    Signature: ols_model_evaluation(data, predictor='', target='')
    Docstring:
    Evaluates a baseline and an OLS model generated from `predictor` and `target`

    Parameters
    ----------
    data : pandas DataFrame
        The dataframe where the predictor and target are stored
    predictor : str, '' by default
        The attribute that will be used to calculate the target variable
    target : str, '' by default
        The target variable
    
    Returns
    -------
    Displays metrics SSE, MSE, RMSE for Baseline and OLS Model
    Displays evaluation of Baseline SSE v. OLS Model SSE
    None
    '''
    
    if predictor == '' or target == '':
        print("Please enter an attribute and a target variable name")
        return None
    
    # Initialize an empty dataframe to store:
    # x, y, baseline predictions, baseline residuals, model predictions, model residuals
    df = pd.DataFrame()
    
    # Create x and y columns to store attribute values and target values
    df['x'] = data[predictor]
    df['y'] = data[target]
    
    # Baseline predictions and baseline residual
    df['baseline_yhat'] = df.y.mean()
    df['baseline_residual'] = df.baseline_yhat - df.y
    
    # Create a OLS model object and fit the data
    model = ols(f'{target} ~ {predictor}', data).fit()
    
    # Model predictions and baseline residual
    df['model_yhat'] = model.predict()
    df['model_residual'] = df.model_yhat - df.y
    
    ols_metrics(df)
    
    return None
    
    
def ols_metrics(df):
    '''
    Signature: ols_metrics(df)
    Docstring:
    Evaluates a Baseline and a OLS Model by calculating: SSE, MSE, RMSE
    
    Parameters
    ----------
    df : pandas DataFrame
        The dataframe of 'ols_model_evaluation'

    Returns
    -------
    Displays metrics SSE, MSE, RMSE for Baseline and OLS Model
    Displays evaluation of Baseline SSE v. OLS Model SSE
    None
    
    '''
    # Calculate Baseline MSE, SSE, RMSE
    baseline_mse = mean_squared_error(df.y,df.baseline_yhat)
    baseline_sse = baseline_mse * len(df.baseline_yhat)
    baseline_rmse = sqrt(baseline_mse)
    
    # Calulate Model MSE, SSE, RMSE
    model_mse = mean_squared_error(df.y, df.model_yhat)
    model_sse = model_mse * len(df.model_yhat)
    model_rmse = sqrt(model_mse)
    
    print("Baseline Metrics")
    print('-' * 15)
    print(f"Basline SSE {baseline_sse:.2f}")
    print(f"Basline MSE {baseline_mse:.2f}")
    print(f"Basline RMSE {baseline_rmse:.2f}")
    
    print("\nModel Metrics")
    print('-' * 15)
    print(f"Basline SSE {model_sse:.2f}")
    print(f"Basline MSE {model_mse:.2f}")
    print(f"Basline RMSE {model_rmse:.2f}")
    
    sse_comparison(baseline_sse, model_sse)
    
    return None
        

def sse_comparison(baseline_sse, model_sse):
    '''
    Signature: sse_comparison(baseline_sse, model_sse)
    Docstring:
    Compares the Baseline SSE and OLS Model SSE.

    Parameters
    ----------
    baseline_sse : float
        The baseline sse from `ols_metrics`
    model_sse : float
        The model sse from `ols_metrics`

    Returns
    -------
    Displays a message of evaluating Baseline SSE v. OLS Model SSE
    None
    '''
    evaluation = baseline_sse/model_sse
    
    if evaluation > 1:
        print("\nOur model beats the baseline!")
        print(f"SSE: {model_sse:.2f} < {baseline_sse:.2f}")
        print(f"Multiple below the baseline: {baseline_sse/model_sse:.2f}")
        print("Let's evaluate the model with a few more tests.")
    else:
        print("Return back to the workbench. This model does not outperform the baseline.")    