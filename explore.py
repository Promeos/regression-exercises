import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def plot_variable_pairs(df):
    '''
    Accepts the telco_churn train set
    Returns all pairwise relationships between attributes
    '''
    
    columns_to_plot = df.select_dtypes(exclude='O').columns.values

    for column in columns_to_plot:
        for pair in columns_to_plot:
            if column != pair:
                sns.regplot(x=column,
                            y=pair,
                            data=df,
                            line_kws={"color": "red"},
                           ).set_title(column + " and " + pair)
                plt.show()


def months_to_years(df):
    df['tenure_years'] = round(df.tenure / 12, 0)
    df = df[['customer_id',
             'monthly_charges',
             'tenure',
             'tenure_years',
             'total_charges'
    ]]
    return df


def monthly_charges_cohorts(df):
    
        df['monthly_charges_cohorts'] = pd.cut(df.monthly_charges,
                                               bins=[18, 38, 58, 78, 98, 118, 138]
                                              )
        return df
    
      
def plot_categorical_and_continuous_cars(categorical_var, continuous_var, df):
    '''
    Accepts
    Returns
    '''
    sns.boxplot(data=df,
                x=categorical_var,
                y=continuous_var
               )
    plt.show()
    
    sns.swarmplot(data=df,
                  x=categorical_var,
                  y=continuous_var
                 )
    plt.show()
    
    sns.violinplot(data=df,
                x=categorical_var,
                y=continuous_var
                  )
    plt.show()
