import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, MinMaxScaler, RobustScaler


def split_my_data(X, y, train_pct=0.75, state=369):
    '''
    Accepts X, y, train_pct
    Returns four dataframes: X_train, X_test, y_train, y_test

    Parameters
    ----------
    X : pandas DataFrame
        Accepts the telco_churn dataframe attributes
        
    y : pandas DataFrame
        Accepts the telco_churn dataframe attributes
        
    train_pct : float, optional, 0.75 by default
        The portion of data that will be in the train set.
        Value must be between 0 and 1.

    random_state : int, optional, 369 by default
        Set the random state of the split to establish
        reproducibility.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size=train_pct,
                                                        random_state=state
                                                        )
    
    return X_train, X_test, y_train, y_test
    

def standard_scaler(train, test):
    '''
    Accepts the train set and test set
    Returns: Scaler Object, scaled_train, scaled_test
    '''
    scaler = StandardScaler()
    scaler, df_scaled_train, df_scaled_test = data_prep(scaler, train, test)
    
    return scaler, df_scaled_train, df_scaled_test


def scale_inverse(scaler, scaled_train, scaled_test):
    '''
    Accepts the train set and test set
    Returns: Scaler Object, scaled_train, scaled_test
    '''
    train_inverse = scaler.inverse_transform(scaled_train)
    test_inverse = scaler.inverse_transform(scaled_test)
    scaler, train, test = data_prep(scaler, train_inverse, test_inverse, inverse=True)
    return scaler, train, test


def uniform_scaler(train, test):
    '''
    Accepts the train set and test set
    Returns: Scaler Object, scaled_train, scaled_test
    '''
    scaler = QuantileTransformer()
    scaler, df_scaled_train, df_scaled_test = data_prep(scaler, train, test)
    return scaler, df_scaled_train, df_scaled_test


def gaussian_scaler(train, test, method='yeo-johnson'):
    '''
    Accepts the train set and test set
    Returns: Scaler Object, scaled_train, scaled_test
    '''
    scaler = PowerTransformer(method=method)
    scaler, df_scaled_train, df_scaled_test = data_prep(scaler, train, test)
    return scaler, df_scaled_train, df_scaled_test


def min_max_scaler(train, test):
    '''
    Accepts the train set and test set
    Returns: Scaler Object, scaled_train, scaled_test
    '''
    scaler = MinMaxScaler()
    scaler, df_scaled_train, df_scaled_test = data_prep(scaler, train, test)
    return scaler, df_scaled_train, df_scaled_test


def iqr_robust_scaler(train, test):
    '''
    Accepts the train set and test set
    Returns: Scaler Object, scaled_train, scaled_test
    '''
    scaler = RobustScaler()
    scaler, df_scaled_train, df_scaled_test = data_prep(scaler, train, test)
    return scaler, df_scaled_train, df_scaled_test


def data_prep(scaler, train, test, inverse=False):
    '''
    Accepts a Scaler Object, train set, test set, and inverse
    Returns a Scaler Object, scaled training data, scaled test data
    
    Parameters
    ----------
    scaler : sklearn.preprocessing object
        Accepts any sklearn.preprocessing object
        
    train : pandas DataFrame
        The training data set
        
    test : pandas DataFrame
        The test data set

    inverse : bool, optional, False by default
        If False: Returns scaler, scaled training data, scaled test data
        If True: Returns scaler, inverse of scaled train data, inverse of scaled test data
        Reverts the scaled data back into original units.
    '''
    if inverse == False: 
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

################################################################################
############################### Master Functions ###############################
# def split_my_data(df, train_pct=0.75, scaled=False, scaler_method='standard'):
#     '''
#     Accepts prepared telco_churn DataFrame
    
#     Returns
#     1. If scaled=False: returns train and test sets
#     2. If scaled=True: returns scaler object, scaled train set, scaled test set
    
#     Parameters
#     ----------
#     df : pandas DataFrame
#         Accepts the prepared telco_churn DataFrame from wrangle_telco()

#     train_pct : float, 0.75 by default
#         The portion of data that will be in the train set.
#         Value must be between 0 and 1.
        
#     scaled : boolean, optional, default is False
#         If False: returns data split into train and test sets.
#         If True: returns a scaler, scaled training set, and scaled test set
        
#     scaler_method : str, optional, default is "standard"
#         Scaler object that is used to scale numeric data.
        
#         acceptable strings:
#         "standard", "inverse", "uniform", "normal", "minmax", "iqrobust"
        
#         Scalers Objects
#         ---------------
#         "standard" returns StandardScaler()
#         "uniform"  returns QuantileTransformer()
#         "normal"   returns PowerTransformer()
#         "minmax"   returns MinMaxScaler()
#         "iqrobust" returns RobustScaler()
    
#     Notes
#     -----
#     https://scikit-learn.org/stable/modules/preprocessing.html
        
#     '''
#     train, test = train_test_split(df, train_size=train_pct, random_state=369)

#     if scaled == False:
#         return train, test
#     elif scaled == True:
#         scaler = preprocessing_scaler(scaler_method)
#         scaler, train_scaled, test_scaled = scale_data(scaler, train, test)
#         return scaler, train_scaled, test_scaled
#     else:
#         return None


# def preprocessing_scaler(scaler_method, gaussian_transformer='yeo-johnson', random=369):
#     '''
#     This function accepts the name of a scaler object (listed below)
#     Returns a data preprocessing object

#     Parameters
#     ----------
#     scaler_method : str
#         Scaler object that is used to scale numeric data.
        
#         acceptable strings:
#         "standard", "inverse", "uniform", "normal", "minmax", "iqrobust"
        
#         Scaler Objects
#         --------------
#         'standard' returns StandardScaler()
#         'uniform'  returns QuantileTransformer()
#         'normal'   returns PowerTransformer()
#         'minmax'   returns MinMaxScaler()
#         'iqrobust' returns RobustScaler()
        
#     gaussian_transformer : str, optional, default is "yeo-johnson"
#         gaussian transformer passed as a argument to PowerTransformer()
        
#         If scaler_method = 'standard' and,
#         gaussian_transformer = "yeo-johnson": returns data scaled using the "yeo-johnson" method
#         gaussian_transformer = "box-cox": returns data scaled using the "box-cox" method
    
#     random : int, optional, default 369
#         The random seed set for QuantileTransformer()
    
#     Notes
#     -----
#     https://scikit-learn.org/stable/modules/preprocessing.html
#     '''
#     if scaler_method == 'standard':
#         scaler = StandardScaler(copy=True,
#                                 with_mean=True,
#                                 with_std=True
#                                )
        
#     elif scaler_method == 'uniform':
#         scaler = QuantileTransformer(n_quantiles=100,
#                                      output_distribution='uniform',
#                                      random_state=random,
#                                      copy=True
#                                     )
        
#     elif scaler_method == 'normal':
#         scaler = PowerTransformer(method=gaussian_transformer,
#                                   standardize=False, 
#                                   copy=True
#                                  )
        
#     elif scaler_method == 'minmax':
#         scaler = MinMaxScaler(copy=True,
#                               feature_range=(0,1)
#                              )
        
#     elif scaler_method == 'iqrobust':
#         scaler = RobustScaler(quantile_range=(25.0,75.0),
#                               copy=True,
#                               with_centering=True,
#                               with_scaling=True
#                              )
#     else:
#         return None
    
#     return scaler


# def scale_data(scaler, train, test):
#     '''
#     This function accepts a scaler object, train set, test set
#     Returns scaler and scaled train and test data as a Pandas DataFrame.
#     '''
#     train = train.drop(columns='customer_id')
#     test = test.drop(columns='customer_id')
#     scaled_train = scaler.fit_transform(train)
#     scaled_test = scaler.transform(test)
    
#     df_scaled_train = pd.DataFrame(scaled_train,
#                                    columns=train.columns.values
#                                   ).set_index([train.index.values])
    
#     df_scaled_test = pd.DataFrame(scaled_test,
#                                   columns=test.columns.values
#                                  ).set_index([test.index.values])
    
#     return scaler, df_scaled_train, df_scaled_test


# def scale_inverse(scaler, scaled_train, scaled_test):
#     '''
#     This function accepts a scaler object, scaled train data, and scaled test data
#     Returns a scaler with original train and test data
#     '''
#     train_inverse = scaler.inverse_transform(scaled_train)
#     test_inverse = scaler.inverse_transform(scaled_test)
    
#     train = pd.DataFrame(train_inverse,
#                          columns=scaled_train.columns.values
#                         ).set_index([scaled_train.index.values])
    
#     test = pd.DataFrame(test_inverse,
#                         columns=scaled_test.columns.values
#                        ).set_index([scaled_test.index.values])
    
#     return scaler, train, test