
import pandas as pd
import numpy as np
from matplotlib import pyplot
# import matplotlib.pyplot as plt
import utils
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

linear_data = pd.read_csv('linear.csv')

linear_data.head(5)

features = np.array(linear_data[['x_1','x_2']])
labels = np.array(linear_data['y'])
utils.plot_points(features, labels)

# C = 1
svm_linear = SVC(kernel = 'linear')
svm_linear.fit(features, labels)
print("Accuracy:", svm_linear.score(features, labels))
utils.plot_model(features, labels, svm_linear)

# C = 0.01
svm_c_001 = SVC(kernel = 'linear', C = 0.01)
svm_c_001.fit(features, labels)
print("C = 0.01")
print("Accuracy:", svm_c_001.score(features, labels))
utils.plot_model(features, labels, svm_c_001)

# C = 100
svm_c_100 = SVC(kernel = 'linear', C = 100)
svm_c_100.fit(features, labels)
print("C = 100")
print("Accuracy:", svm_c_100.score(features, labels))
utils.plot_model(features, labels, svm_c_100)

# C = 50
svm_c_50 = SVC(kernel = 'linear', C = 100)
svm_c_50.fit(features, labels)
print("C = 50")
print("Accuracy:", svm_c_50.score(features, labels))
utils.plot_model(features, labels, svm_c_50)


# load one circle data

circular_data = pd.read_csv('one_circle.csv')

circular_data.head(5)

features = np.array(circular_data[['x_1','x_2']])
labels = np.array(circular_data['y'])
utils.plot_points(features, labels)

# Polynomial kernel of degree 2
svm_degree_2 =  SVC(kernel = 'poly', degree = 2)
svm_degree_2.fit(features, labels)
print("Polynomial kernel of degree 2")
print("Accuracy:", svm_degree_2.score(features, labels))
utils.plot_model(features, labels, svm_degree_2)

# Polynomial kernel of degree 4
svm_degree_4 =  SVC(kernel = 'poly', degree = 4)
svm_degree_4.fit(features, labels)
print("Polynomial kernel of degree 4")
print("Accuracy:", svm_degree_4.score(features, labels))
utils.plot_model(features, labels, svm_degree_4)


# load two circles data

two_circles_data = pd.read_csv('two_circles.csv')

two_circles_data.head(5)

features = np.array(circular_data[['x_1','x_2']])
labels = np.array(circular_data['y'])
utils.plot_points(features, labels)

# Gamma = 0.1
svm_gm_01 =  SVC(kernel = 'rbf', gamma = 0.1)
svm_gm_01.fit(features, labels)
print("RBF with Gamma as 0.1")
print("Accuracy:", svm_gm_01.score(features, labels))
utils.plot_model(features, labels, svm_gm_01)

# Gamma = 1
svm_gm_1 =  SVC(kernel = 'rbf', gamma = 1)
svm_gm_1.fit(features, labels)
print("RBF with Gamma as 1")
print("Accuracy:", svm_gm_1.score(features, labels))
utils.plot_model(features, labels, svm_gm_1)

# Gamma = 10
svm_gm_10 =  SVC(kernel = 'rbf', gamma = 10)
svm_gm_10.fit(features, labels)
print("RBF with Gamma as 10")
print("Accuracy:", svm_gm_10.score(features, labels))
utils.plot_model(features, labels, svm_gm_10)

# Gamma = 10
svm_gm_100 =  SVC(kernel = 'rbf', gamma = 100)
svm_gm_100.fit(features, labels)
print("RBF with Gamma as 100")
print("Accuracy:", svm_gm_100.score(features, labels))
utils.plot_model(features, labels, svm_gm_100)

# SVM Best Parameter Search

svm_parameters = {'kernel' : ['rbf'], 'C' : [0.01, 0.1, 1, 10, 100], 'gamma' : [0.01, 0.1, 1, 10, 100]}
# svm_parameters = {'kernel':['rbf'],
#                 'C': [0.01, 0.1, 1, 10, 100],
#                'gamma': [0.01, 0.1, 1, 10, 100]}

svm = SVC()

svm_gs = GridSearchCV(estimator = svm, param_grid = svm_parameters)

svm_gs.fit(features, labels)

svm_best = svm_gs.best_estimator_
svm_best.score(features, labels)
svm_best 
# Should ideally be done on validation set

## Bayesian Optimization can be tried to search for a good combination 
## of parameters that will optimize the model


## utils.py 
# Some functions to plot our points and draw the lines

def plot_points(features, labels):
    X = np.array(features)
    y = np.array(labels)
    spam = X[np.argwhere(y==1)]
    ham = X[np.argwhere(y==0)]
    pyplot.scatter([s[0][0] for s in spam],
                   [s[0][1] for s in spam],
                   s = 35,
                   color = 'cyan',
                   edgecolor = 'k',
                   marker = '^')
    pyplot.scatter([s[0][0] for s in ham],
                   [s[0][1] for s in ham],
                   s = 25,
                   color = 'red',
                   edgecolor = 'k',
                   marker = 's')
    pyplot.xlabel('x_1')
    pyplot.ylabel('x_2')
    pyplot.legend(['label 1','label 0'])

def plot_model(X, y, model):
    X = np.array(X)
    y = np.array(y)
    plot_step = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    pyplot.contour(xx, yy, Z,colors = 'k',linewidths = 3)
    plot_points(X, y)
    pyplot.contourf(xx, yy, Z, colors=['red', 'blue'], alpha=0.2, levels=range(-1,2))
    pyplot.show()
