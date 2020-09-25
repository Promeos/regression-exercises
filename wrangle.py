import pandas as pd
from acquire import get_telco_data


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

    return df


