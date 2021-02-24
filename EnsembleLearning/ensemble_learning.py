import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def get_unique(x):
    '''
    Returns all unique values of x as a list
    '''
    return list(set(x))



def entropy(y, fn_log = np.log2):
    '''
    Calculates entropy
    
    Inputs:
    y      -- an array of labels
    fn_log -- the log function used when calculating entropy
    
    Returns:
    H -- entropy for all given labels
    '''
    
    # get unique values
    y_unique = list(set(y))
    
    if len(y_unique) < 2:
        return 0
    
    # get total number of values
    y_total = len(y)
    
    # compute ratios (p- p+)
    ratio=[np.sum(y == i) / y_total for i in y_unique]
    
    # compute entropy
    H = np.sum([-p * fn_log(p) for p in ratio])
    
    return H



def majority_error(y):
    '''
    Calculates majority error for the labels contained in y
    '''
    
    # get unique values
    y_unique = list(set(y))
    
    # get total number of values
    y_total = len(y)
    
    # compute ratios (p- p+)
    ratio=[np.sum(y == i) / y_total for i in y_unique]
    
    # compute ME
    H = 1 - max(ratio)
    
    return H



def gini_index(y):
    '''
    Calculates Gini Index for the labels contained in y
    '''
    
    # get unique values
    y_unique = list(set(y))
    
    # get total number of values
    y_total = len(y)
    
    # compute ratios (p- p+)
    ratio=[np.sum(y == i) / y_total for i in y_unique]
    
    # compute Gini index
    H = np.sum([p**2 for p in ratio])
    
    G = 1 - H
    
    return G



def compute_gain(x, y, fn_gain=entropy, round_values=False):
    '''
    Calculates the information gain
    
    Inputs:
    x            -- data shape (n, a) where n is number of samples, and a is number of attributes
    y            -- labels shape (n,)
    fn_gain      -- function to compute the gain
    round_values -- if true information gain is rounded to 10 decimal places
    
    Returns:
    Ht -- list of information gain for each attribute in x
    '''
    
    # get heuristic for all data
    Hy = fn_gain(y)
    
    # initialize to store data
    Ht=[]
    
    # iterate over all attributes in x
    for col in range(x.shape[1]):
        
        # select data for current attribute
        xi = x[:,col]

        # get unique values for x
        x_unique = list(set(xi))

        # get total values for x
        x_total = len(xi)

        # initialize lists to store output
        w=[] # weights
        h=[] # heuristics

        # iterate through all xi_unique
        for val in x_unique:

            # get index where xi = current value
            idx = np.where(xi == val)

            # subset y
            y_sub = y[idx]

            w.append(len(y_sub) / x_total) # append weight (ratio)
            h.append(fn_gain(y_sub)) # append entropy

        # convert to np arrays
        w = np.array(w)
        h = np.array(h)

        # sum element-wise product to weight
        Ht.append(np.sum(w * h))
    
    # for final output perform subtraction 
    Ht = np.array(Ht)
    Ht = Hy - Ht
    
    # round output
    if round_values == True:
        Ht = Ht.round(10)
    
    return Ht



class Node(object):
    '''
    Node object used to construct a tree
    
    Attributes:
    key      -- stores the index of the attribute used to partition data
    leaf     -- bool stating whether the node is a leaf
    label    -- label of the leaf node
    branches -- list of all the branches from the node
    children -- list of the nodes children
    
    Methods:
    add_child -- method to append a child to the parent nodes children attribute
    '''
    def __init__(self, key=None, leaf=False, label=None, branches=None, most_common=None):
        
        self.key = key
        self.leaf = leaf
        self.label = label
        self.branches = branches
        self.most_common = most_common
        self.children=[]
        
    def add_child(self, obj):
        '''
        Method to append a child to the parent nodes children attribute
        '''
        self.children.append(obj)

        
        
def id3(X, y, function=entropy, A=[], Av={}, depth=-1, max_depth=np.float('inf'), prev_common=None):
    '''
    ID3 algorithm used to construct a decision tree through recursion
    
    Inputs:
    X         -- data of shape (n, a) where n is number of samples and a is number of attributes
    y         -- lables for data of shape (n,)
    function  -- function to be used when computing gain
    A         -- list of all available attributes to partition the data
    Av        -- dictionary of all branches from a given attribute
    depth     -- counter to track the depth of the tree
    max_depth -- integer specifying how deep a tree is allowed to grow
    
    Returns:
    root -- root node from a pass through the algorithm. With recursion this creates a tree
    '''
    # increase level of depth every time the function is called
    depth += 1
    
    # check that X and y are not empty
    # If empty return leaf with previous most common label
    if len(y) == 0:
        return Node(leaf=True, label=prev_common)
    
    # get all unique values of y and the most common value of y
    y_unique = get_unique(y)
    y_common = np.bincount(y).argmax()
    
    # if all examples have same label return a leaf node with label
    if len(y_unique) < 2:
        return Node(leaf=True, label=y_unique[0])
    
    # if the max depth has been exceeded return a leaf node with the most common label
    elif depth >= max_depth:
        return Node(leaf=True, label=y_common)
    
    else:
        # find attribute that best splits X with information gain
        # note X is a subset based on A so idx needs to reference A to get true index (key)
        idx = compute_gain(X[:, A], y, fn_gain=function).argmax()
        
        # create a node for best attribute A
        key = A[idx] # best attibute
        root = Node(key=key, branches=np.array(Av[key]), most_common=y_common) # root node**
        
        # get unique values of A (branches)
        child_values = X[:, key]
        
        # remove current best attribute (key) from the list A for A_next
        A_next = A[A != key].copy()
        
        # loop over each subset of A (each branch)
        for i, v in enumerate(Av[key]):
            
            # subset data
            # get index where child values == current branch v
            child_idx = np.where(child_values == v)[0]
                
            # subset data
            X_next = X[child_idx].copy() # subset X
            y_next = y[child_idx].copy() # subset y
            
            # call id3 for next data (X_next, y_next, A_next) going one level deeper
            # note the output of id3 is added as a child to the root node created above **
            root.add_child(id3(X_next, y_next, function=function, A=A_next, Av=Av, depth=depth, max_depth=max_depth, prev_common=y_common))
            
        return root

    
    
def predict_single(t, tree, verbose=False):
    '''
    Travels through tree to a leaf node and returns the leafs label
    
    Inputs:
    t       -- single example of data
    tree    -- decision tree created with id3 algorithm and Node class
    verbose -- bool stating whether to print information about the prediction
    
    Returns:
    tree.label -- single predicted label for current example
    '''
    
    count=0
    while True:
        # return label if leaf node
        if tree.leaf:
            if verbose:
                print(f"Return: {tree.label}\n")
            return tree.label

        # get first attribute key
        t_key = t[tree.key]
        # find corresponding child
        if t_key in tree.branches:
            idx = np.where(tree.branches == t_key)[0][0]
            
            if verbose:
                print(f"{count:02} - key: {tree.key}, child: {idx}, branch: {tree.branches[idx]}")
        else:
            
            if verbose:
                print(f"Return: {tree.most_common} from most common\n")
            return tree.most_common
        
        # update tree (travel toward leaf)
        tree = tree.children[idx]
        count+=1

        
        
def predict_many(t, tree, verbose=False):
    '''
    Uses the previously defined predict_single function and applies it to multiple examples
    
    Inputs:
    t       -- array example of data of shape (n, a) where n is number of samples and a is number of attributes
    tree    -- decision tree created with id3 algorithm and Node class
    verbose -- bool stating whether to print information about the prediction
    
    Returns:
    y_pred -- array of predicted labels of shape (n,)
    '''
    # initialize list to store predictions
    y_pred=[]
    
    # loop over each example of data and make prediction
    for i in range(t.shape[0]):
        y_pred.append(predict_single(t[i], tree, verbose=verbose))
        
    return np.array(y_pred)



def get_medians(X):
    '''
    Calculates attribute-wise median of numeric data
    
    Input:
    X -- data of shape (n, a) where n is number of samples and a is number of attributes
    
    Returns: 
    medians -- dictionary of {attribute index: attribute mean} for all numeric attributes
    '''
    medians={}
    for j in range(X.shape[1]):
        try:
            med = np.median(X[:, j])
            medians[j] = med
        except:
            pass
        
    return medians



def accuracy(y_pred, y):
    '''
    Calculates accuracy of predictions
    '''
    # check that prediction and truth are of equal length
    assert len(y_pred) == len(y), 'Inputs must be of equal length'
    
    return np.sum(y_pred == y)/len(y)



class DecisionTree(object):
    '''
    Class that combines previously defined functions and objects for convenience
    
    Attributes:
    _function  -- function that should be used to calculate gain
    _max_depth -- max depth the tree is allowed to grow
    _medians   -- dictionary containing medians of all numeric attributes in X
    _tree      -- decision tree created with id3 algorithm
    
    Mehtods:
    fit     -- creates decision tree (_tree)
    predict -- makes prediction from decision tree
    '''
    
    def __init__(self, function=entropy, max_depth=np.float('inf')):
        
        # initialize variables
        self._function = function
        self._maxDepth = max_depth
        self._medians = None
        self._tree = None
        
    def fit(self, X, y):
        
        # copy data into class so original data is not changed
        self._X = X.copy()
        self._y = y.copy()
        
        # process numeric data by above median True, below or equal to median False
        self._medians = get_medians(self._X)
        for k in self._medians.keys():
            self._X[:, k] = self._X[:, k] > self._medians[k]
        
        # get list of attributes (A) and all branches (Av)
        # initialized to avoid conflicts in scope
        A, Av = [], {}
        
        # create A numbering all attributes of X
        A = np.arange(self._X.shape[1])
        
        # if first call create a dictionary Av of all branches associated with each attribute in A
        for ai in A:
                Av[ai] = sorted(get_unique(self._X[:,ai]))
        
        # fit tree with id3 algorithm
        self._tree = id3(self._X, self._y, function=self._function, A=A, Av=Av, max_depth=self._maxDepth)
        
    def predict(self, X, verbose=False):
        
        # copy data into class so original data is not changed
        self._Xpred = X.copy()
        
        # reshape if single example
        if len(self._Xpred.shape) < 2:
            self._Xpred = self._Xpred.reshape(1, -1)
        
        # process numeric data
        if len(self._medians) > 0:
            for k in self._medians.keys():
                self._Xpred[:, k] = self._Xpred[:, k] > self._medians[k]

        # return predictions
        return predict_many(self._Xpred, self._tree, verbose=verbose)
    
    
    
def dtree_accuracy(X_train, y_train, X_test, y_test, functions, depths):
    '''
    Analyzes the decision tree created with various functions and depths
    
    Inputs:
    X_train   -- training data of shape (n, a)
    y_train   -- training labels of shape (n,)
    X_test    -- test data of shape (n, a)
    y_test    -- test labels of shape (n,)
    functions -- list of functions to be used in gain computation
    depths    -- list of depths to be used as max_depth in the trees
    
    Returns:
    results_train -- dictionary containing results for different function and depth combinations for training data
    results_test  -- dictionary containing results for different function and depth combinations for test data
    '''
    
    # initialize dictionaries to store results
    results_train = {f.__name__:[] for f in functions}
    results_test = {f.__name__:[] for f in functions}
    
    # loop over all functions and depths
    for f in functions:
        for d in depths:
            
            # create and fit model
            model = DecisionTree(function=f, max_depth=d)
            model.fit(X_train, y_train)
            
            # make predictions and update results for training data
            y_pred = model.predict(X_train)
            results_train[f.__name__].append(accuracy(y_pred, y_train))
            
            # make predictions and update results for test data
            y_pred = model.predict(X_test)
            results_test[f.__name__].append(accuracy(y_pred, y_test))
            
            del(model)
            
    results_train['depth']=depths
    results_test['depth']=depths
    
    return results_train, results_test