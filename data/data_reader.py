import os
import numpy as np
import pandas as pd

def data_reader(data):
    '''
    This function reads in data split into test and training sets
    
    Input:
    data -- string specifying what folder to use to read in data
    
    Output
    (X_train, y_train), (X_test, y_test), desc
    '''
    
    colnames = {'car': ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label'],
               'bank': ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label'],
               'bank-note': ['variance', 'skewness', 'curtosis', 'entropy', 'y'],
               'concrete': ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'Slump']}
    
    data_parent = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(data_parent, data)
    
    assert os.path.isdir(data_path), f"Invalid data selection. Valid selections include: {sorted(colnames.keys())}"
    
    df_train = pd.read_csv(os.path.join(data_path, 'train.csv'), header=None, names=colnames[data])
    df_test = pd.read_csv(os.path.join(data_path, 'test.csv'), header=None, names=colnames[data])
    
    if (data == 'car') or (data == 'bank'):
        
        # factorize train labels
        df_train['label'], legend = df_train['label'].factorize()
        
        # create label legend
        legend_dict={}
        for i, item in enumerate(legend):
            legend_dict[item]=i
        
        # factorize test labels
        df_test['label'] = df_test['label'].apply(lambda x:legend_dict[x])
        
        # get train data
        X_train = df_train.drop('label', axis=1).values
        y_train = df_train['label'].values

        # get test data
        X_test = df_test.drop('label', axis=1).values
        y_test = df_test['label'].values
        
        # record description
        desc = {'columns': colnames[data], 'labels': legend_dict}
        
    elif data == 'bank-note':
        
        # process labels
        df_train['y'] = df_train['y'].apply(lambda x: -1 if x<1 else 1)
        df_test['y'] = df_test['y'].apply(lambda x: -1 if x<1 else 1)
        
        # get train data
        X_train = df_train.drop('y', axis=1).values
        y_train = df_train['y'].values
        
        # get test data
        X_test = df_test.drop('y', axis=1).values
        y_test = df_test['y'].values
        
        # record description
        desc = {'columns': colnames[data]}
        
    elif data == 'concrete':
        
        X_train = df_train.drop('Slump', axis=1).values
        y_train = df_train['Slump'].values
        
        X_test = df_test.drop('Slump', axis=1).values
        y_test = df_test['Slump'].values
        
        # record description
        desc = {'columns': colnames[data]}
        
    return (X_train, y_train), (X_test, y_test), desc