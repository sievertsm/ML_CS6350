import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def accuracy(y_pred, y):
    '''
    Calculates accuracy of predictions
    '''
    # check that prediction and truth are of equal length
    assert len(y_pred) == len(y), 'Inputs must be of equal length'
    
    return np.sum(y_pred == y)/len(y)


def most_common_str(df, exclude=None):

    most_common={}

    for c in df.columns:
        if df[c].dtype == 'object':
            if exclude:
                mode = df[df[c] != 'unknown'][c].mode().values[0]
            else:
                mode = df[c].mode().values[0]

            most_common[c] = mode
                
    return most_common



def replace_with_common(df, common_dict, to_replace='unknown'):
    
    df_update = df.copy()
    
    for c in common_dict.keys():
        df_update[c].replace(to_replace=to_replace, value=common_dict[c], inplace=True)
        
    return df_update