import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, MinMaxScaler, RobustScaler


def data_split(df, test_pct=0.25, scaled=False, scaler_method='standard'):
    '''
    Accepts prepared telco_churn DataFrame
    
    Multiple Return Options:
    1. If scaled=False: returns train and test sets
    2. If scaled=True: returns scaler object, scaled train set, scaled test set
    
    Parameters
    ----------
    df: pandas DataFrame
        Accepts the prepared telco_churn DataFrame from wrangle_telco()

    test_pct: float, 0.25 by default
        The portion of data that will be in the test set.
        Value must be between 0 and 1.
        
    scaled : boolean, optional, default is False
        If False, returns data split into train and test sets.
        If True, returns a scaler, scaled training set, and scaled test set
        
    scaler_method : str, optional, default is "standard"
        Scaler object that is used to scale numeric data
        scaler objects
        "standard", "inverse", "uniform", "normal", "minmax", "iqrobust"
        
        Scalers objects
        ---------------
        "standard" returns StandardScaler()
        "uniform"  returns QuantileTransformer()
        "normal"   returns PowerTransformer()
        "minmax"   returns MinMaxScaler()
        "iqrobust" returns RobustScaler()
    
    Notes
    -----
    https://scikit-learn.org/stable/modules/preprocessing.html
        
    '''
    train, test = train_test_split(df, test_size=test_pct, random_state=369)

    if scaled == False:
        return train, test
    elif scaled == True:
        scaler = preprocessing_scaler(scaler_method)
        scaler, train_scaled, test_scaled = scale_data(scaler, train, test)
        return scaler, train_scaled, test_scaled
    else:
        return None


def preprocessing_scaler(scaler_method, gaussian_transformer='yeo-johnson',random=369):
    '''
    This function accepts a string and returns a data preprocessing object

    scaler_method : str, optional, default is 'standard'
        Scaler object that is used to scale numeric data
        scaler objects
        'standard', 'inverse', 'uniform', 'normal', 'minmax', 'iqrobust'
        
        Scaler objects
        --------------
        'standard' returns StandardScaler()
        'uniform'  returns QuantileTransformer()
        'normal'   returns PowerTransformer()
        'minmax'   returns MinMaxScaler()
        'iqrobust' returns RobustScaler()
        
    Notes
    -----
    https://scikit-learn.org/stable/modules/preprocessing.html
    '''
    if scaler_method == 'standard':
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    elif scaler_method == 'uniform':
        scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=random, copy=True)
    elif scaler_method == 'normal':
        scaler = PowerTransformer(method=gaussian_transformer, standardize=False, copy=True)
    elif scaler_method == 'minmax':
        scaler = MinMaxScaler(copy=True, feature_range=(0,1))
    elif scaler_method == 'iqrobust':
        scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True)
    else:
        return None
    
    return scaler


def scale_data(scaler, train, test):
    '''
    This function accepts a scaler object, train set, test set
    Returns scaler and scaled train and test data as a Pandas DataFrame.
    '''
    train = train.drop(columns='customer_id')
    test = test.drop(columns='customer_id')
    scaled_train = scaler.fit_transform(train)
    scaled_test = scaler.transform(test)
    
    df_scaled_train = pd.DataFrame(scaled_train,
                                columns=train.columns.values
                                ).set_index([train.index.values])
    
    df_scaled_test = pd.DataFrame(scaled_test,
                                 columns=test.columns.values
                                 ).set_index([test.index.values])
    
    return scaler, df_scaled_train, df_scaled_test


def scale_inverse(scaler, scaled_train, scaled_test):
    '''
    This function accepts a scaler object, scaled train data, and scaled test data
    Returns a scaler with original train and test data
    '''
    train_inverse = scaler.inverse_transform(scaled_train)
    test_inverse = scaler.inverse_transform(scaled_test)
    
    train = pd.DataFrame(train_inverse,
                        columns=scaled_train.columns.values
                        ).set_index([scaled_train.index.values])
    
    test = pd.DataFrame(test_inverse,
                        columns=scaled_test.columns.values
                        ).set_index([scaled_test.index.values])
    
    return scaler, train, test