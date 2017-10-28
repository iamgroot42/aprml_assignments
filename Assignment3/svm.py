# coding: utf-8

from sklearn import svm, metrics
from sklearn.linear_model import SGDClassifier
import numpy as np
from load_data import load


def incremental_SVM(X, Y):
    clf = svm.LinearSVC()
    clf.fit(X, Y)
    return clf


def incremental_SGD(X, Y, loss):
    sgd = SGDClassifier(loss=loss, penalty="l2")
    labels = np.unique(Y)
    for i in range(X.shape[0]):
        point_x = X[i]
        point_y = Y[i]
        sgd.partial_fit([point_x], [point_y], classes=labels)
    return sgd

(X_train, Y_train), (X_test, Y_test), cross_val_indices = load()
X_train = X_train.reshape((X_train.shape[0],) + (np.prod(X_train.shape[1:]),))
X_test = X_test.reshape((X_test.shape[0],) + (np.prod(X_test.shape[1:]),))
Y_train = np.argmax(Y_train, axis=1)
Y_test = np.argmax(Y_test, axis=1)

losses = ["hinge", "log", "modified_huber", "squared_hinge"]
for loss in losses:
    mean_error = 0
    for i, val in enumerate(cross_val_indices):
        sgd = incremental_SGD(X_train[val], Y_train[val], loss)
	error = sgd.score(X_test, Y_test)
        print("For validation %d using %s loss" % (i, loss), error)
	mean_error += error
    print("Mean error for %s loss was %f" % (loss, mean_error / (1.0*len(cross_val_indices))))


mean_error = 0
for i, val in enumerate(cross_val_indices):
	sgd = incremental_SVM(X_train[val], Y_train[val])
	error = sgd.score(X_test, Y_test)
 	mean_error += error
print("Mean error for linear SVM was %f" % (mean_error / (1.0*len(cross_val_indices))))

