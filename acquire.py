import env
import os
import pandas as pd


def get_connection(db, user=env.user, host=env.host, password=env.password):
    '''
    Returns a formatted url to access a SQL database.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


def get_mall_customer_data():
    '''
    This funtion reads in the mall customer data from the Codeup db into
    a df, writes it to a csv file, and returns the df.
    '''
    sql_query = "SELECT * FROM customers;"
    file = 'mall_customers.csv'
    
    if os.path.isfile(file):
        return pd.read_csv('mall_customers.csv')
    else:
        df = pd.read_sql(sql_query, get_connection('mall_customers'))
        df.to_csv('mall_customers.csv')
        return df


def get_titanic_data():
    '''
    Returns the titanic dataset.
    '''
    file = 'titanic.csv'

    if os.path.isfile(file):
        return pd.read_csv('titanic.csv')
    else:
        df = pd.read_sql('select * from passengers;', get_connection('titanic_db'))
        df.to_csv('titanic.csv', index=False)
        return df


def get_iris_data():
    '''
    Returns the iris dataset w/ iris species labels.
    '''
    file = 'iris.csv'

    if os.path.isfile(file):
        return pd.read_csv('iris.csv')
    else:
        df = pd.read_sql("""
                        select *from measurements
                        join species using(`species_id`);
                        """,
                        get_connection('iris_db'))
        df.to_csv('iris.csv', index=False)
        return df


