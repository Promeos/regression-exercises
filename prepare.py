# noinspection PyUnresolvedReferences
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from acquire import get_titanic_data, get_iris_data


def prep_iris(df=get_iris_data()):
    """
    prep_iris accepts the iris dataset and returns a transformed iris dataset
    for exploratory analysis.
    type(df) >>> pandas.core.frame.DataFrame
    """
    # Drop columns of redundant data or 'index-like'/ordinal row.
    df.drop(columns=['species_id', 'measurement_id'], inplace=True)

    # Rename species_name to be concise.
    df.rename(columns={'species_name': 'species'}, inplace=True)

    # Create dummy variables for our targets - 0 0 represents 'species_setosa'
    encoded_species = pd.get_dummies(df.species, drop_first=True)

    # Add the encoded target names as columns to the dataframe.
    df = pd.concat([df, encoded_species], axis=1)

    return df


def prep_titanic(df=get_titanic_data()):
    """
    prep_titanic accepts the titanic dataset and returns a transformed titanic dataset
    for exploratory analysis.
    type(df) >>> pandas.core.frame.DataFrame
    """
    # Drop missing values in the embarked column.
    # This removes missing values in embark_town as well.
    # df.dropna(how='any', subset=['embarked'], inplace=True)

    # Throw the deck overboard because there are too many missing values.
    df.drop(columns=['deck'], inplace=True)

    # Create dummy variables for our targets.
    encoded_embarked = pd.get_dummies(df.embark_town,
                                      drop_first=True)

    encoded_class = pd.get_dummies(df['class'],
                                   drop_first=True)

    encoded_sex = pd.get_dummies(df.sex,
                                 drop_first=True)

    df = df.select_dtypes(exclude='O')
    # Scale numerical columns using MinMaxScalar()
    # scalar = MinMaxScaler()

    # Use `.transform_fit` on the scalar object to fit and transform the data.
    # Assign directly to 'age' and 'fare' columns.
    # df[['age', 'fare']] = scalar.fit_transform(df[['age', 'fare']])

    # Add the encoded target names as columns to the dataframe.
    df = pd.concat([df,
                    encoded_embarked,
                    encoded_class,
                    encoded_sex], axis=1)

    return df


def prep_mall_data(df, modeling=False):
    '''
    This function accepts the mall customers dataframe, and adds
    an encoded column for gender. Optional argument 'modeling' to split data.
    Returns either a dataframe or train, validate, and test data for EDA and Modeling
    
    Drop: customer_id and gender before you split the data for machine
    learning. The algos like numbers, not strings.
    '''
    df['is_female'] = (df.gender == 'Female').astype('int')
    
    df = df[[
    'customer_id',
    'gender',
    'is_female',
    'age',
    'annual_income',
    'spending_score'
    ]]
    
    if modeling == False:
        return df
    elif modeling == True:
        train_validate, test = train_test_split(df, test_size=.15)
        train, validate = train_test_split(train_validate, test_size=.15)
        return train, validate, test
    else:
        return None

