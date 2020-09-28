import pandas as pd
from acquire import get_telco_data
from sklearn.model_selection import train_test_split

def wrangle_telco():
    '''
    This function acquires/loads `telco_churn` data from a SQL database using SQL and Pandas. Cleans the acquired
    data and returns a dataframe to be used in EDA and Modeling.
    '''

    df = get_telco_data()
    df['total_charges'] = df.total_charges.str.strip()
    df.total_charges.replace("", 0, inplace=True)
    df['total_charges'] = df.total_charges.astype('float')
    df = df[['customer_id', 'tenure', 'monthly_charges', 'total_charges']]

    train, validate, test = split_data(df)
    return train, validate, test

def split_data(df):
    '''
    Accepts the telco_churn dataset.
    Returns the telco_churn split into train, validate, and test sets.
    '''
    train_validate, test = train_test_split(df, test_size=.15)
    train, validate = train_test_split(train_validate, test_size=.15)
    return train, validate, test


