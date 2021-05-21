import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data.data_reader import data_reader

from DecisionTree.decision_tree import DecisionTree
from EnsembleLearning.ensemble_learning import Adaboost, Bagging, RandomForest
from LinearModel.linear_regression import LinearRegression
from Perceptron.perceptron import Perceptron
from SVM.svm import SVM
from LogisticRegression.logistic_regression import LogisticRegression
from NeuralNetwork.ann import ArtificialNeuralNet

def get_metric(y_true, y_pred, metric='accuracy'):
    
    if metric=='accuracy':
        acc = np.sum(y_true == y_pred) / len(y_true)
        return acc
    elif metric=='error':
        err = np.sum(y_true != y_pred) / len(y_true)
        return err
    
    
def example_model(model, metric='accuracy'):
    
    model_name = str(model.__class__)[:-2].split('.')[-1]
    
    choose_data = {'DecisionTree': 'bank',
                  'Adaboost': 'bank',
                  'Bagging': 'bank',
                  'RandomForest': 'bank',
                  'LinearRegression': 'concrete',
                  'Perceptron': 'bank-note',
                  'SVM': 'bank-note', 
                  'LogisticRegression': 'bank-note',
                  'ArtificialNeuralNet': 'bank-note',
                  }

    (X_train, y_train), (X_test, y_test), desc = data_reader(data=choose_data[model_name])
    model.fit(X_train, y_train)
    
    if model_name == 'LinearRegression':
        from LinearModel.linear_regression import mean_square_loss
        metric = 'Mean Square Loss'
        train_metric = mean_square_loss(y_train, model.predict(X_train))
        test_metric = mean_square_loss(y_test, model.predict(X_test))
        
    else:
        train_metric = get_metric(y_train, model.predict(X_train), metric=metric)
        test_metric = get_metric(y_test, model.predict(X_test), metric=metric)
    
    print(f"\n{model_name} {metric.capitalize()}")
    print(f"Train {metric.capitalize()}: {round(train_metric, 4)}")
    print(f"Test {metric.capitalize()}:  {round(test_metric, 4)}\n")
    
def run_example():
    my_parser = argparse.ArgumentParser()

    my_parser.add_argument('-model', action='store', choices=['decisiontree', 
                                                              'adaboost', 
                                                              'bagging', 
                                                              'randomforest', 
                                                              'linearregression', 
                                                              'perceptron', 
                                                              'svm', 
                                                              'logisticregression', 
                                                              'ann'], required=True)

    my_parser.add_argument('-value', action='store', default='accuracy', choices=['accuracy', 'error'])

    args = my_parser.parse_args()

    metric = args.value
    model_select = args.model

    model_dict = {'decisiontree': DecisionTree(), 
                  'adaboost': Adaboost(), 
                  'bagging': Bagging(), 
                  'randomforest': RandomForest(), 
                  'linearregression': LinearRegression(tol=1e-6), 
                  'perceptron': Perceptron(), 
                  'svm': SVM(), 
                  'logisticregression': LogisticRegression(), 
                  'ann': ArtificialNeuralNet(input_dim=4, hidden_dim=[10, 10])}

    model = model_dict[model_select]

    example_model(model, metric=metric)
    
if __name__=='__main__':
    run_example()