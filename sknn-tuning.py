

import numpy as np
import scipy as sp 
import pandas as pd
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy import stats
from sklearn.cross_validation import train_test_split
from sknn.mlp import  Layer, Regressor, Classifier as skClassifier


# Load data
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv ' , sep = ';')
X = df.drop('quality' , 1).values # drop target variable
   

y1 = df['quality'].values # original target variable
y = y1 <= 5 # new target variable: is the rating <= 5?

# Split the data into a test set and a training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print X_train.shape


# you might want to relax the parameter space in case of computational load. 
max_net = skClassifier(layers= [Layer("Rectifier",units=10),
                                       Layer("Rectifier",units=10),
                                       Layer("Rectifier",units=10),
                                       Layer("Softmax")])
params={'learning_rate': sp.stats.uniform(0.001, 0.05,.1),
        'hidden0__units': sp.stats.randint(4, 20),
        'hidden0__type': ["Rectifier"],
        'hidden1__units': sp.stats.randint(4, 20),
        'hidden1__type': ["Rectifier"],
        'hidden2__units': sp.stats.randint(4, 20),
        'hidden2__type': ["Rectifier"],
        'batch_size':sp.stats.randint(10,1000),
        'learning_rule':["adagrad","rmsprop","sgd"]}
max_net2 = RandomizedSearchCV(max_net,param_distributions=params,n_iter=25,cv=3,scoring='accuracy',verbose=100,n_jobs=1,\
                             pre_dispatch=None)
model_tuning=max_net2.fit(X_train,y_train)



print "best score %s" % model_tuning.best_score_
print "best parameters %s" % model_tuning.best_params_



                           #                       **********
                           #                     **********
                           #               **************
                           #         **********************
                           #       **************************
                           #       **  ************************
                           #   **********************************
                           # **************            ************
                           #         ****************      ********
                           #               ******    ****    ********
                           #                 ******    ****  ********
                           #                     ****    **  ********
                           #                             **  ********
                           #                             **  ******
                           #               ******      **********
                           #                 ******  **********
                           #                   ************
                           #                   ********
                           #                 ******
                           #               ******
