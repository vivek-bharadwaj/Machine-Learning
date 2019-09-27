"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd


###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################

    err = np.mean(np.abs(np.subtract(np.dot(X, w), np.array(y, np.float64))))
    return err


###### Q1.2 ######
def linear_regression_noreg(X, y):
    """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
    #####################################################
    #	TODO 2: Fill in your code here #
    #####################################################
    w = None
    x_transpose_x = np.matmul(np.transpose(X), X)
    inv_xtx = np.linalg.inv(x_transpose_x)
    x_transpose_y = np.matmul(np.transpose(X), y)
    w = np.matmul(inv_xtx, x_transpose_y)
    return w


###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    x_transpose_x = np.dot(np.transpose(X), X)
    eigen_values = np.linalg.eigvals(x_transpose_x)
    min_ev = abs(min(eigen_values))
    new_min_ev = min_ev
    converted_X = x_transpose_x
    if min_ev < 10e-5:
        # non-invertible matrix. Keep adding 10e-1 * I
        while new_min_ev < 10e-5:
            converted_X = np.add(x_transpose_x, np.identity(len(x_transpose_x)) * 0.1)
            new_min_ev = abs(min(np.linalg.eigvals(converted_X)))
    inv_converted_X = np.linalg.inv(converted_X)
    x_transpose_y = np.dot(np.transpose(X), y)
    w = np.dot(inv_converted_X, x_transpose_y)
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 4: Fill in your code here #
    #####################################################
    x_transpose_x = np.dot(np.transpose(X), X)
    regularized_x_transpose_x = np.add(x_transpose_x, np.identity(len(x_transpose_x)) * lambd)
    inv_x = np.linalg.inv(regularized_x_transpose_x)
    x_transpose_y = np.dot(np.transpose(X), y)
    w = np.dot(inv_x, x_transpose_y)
    return w


###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################		
    bestlambda = None
    lambdA = round(10 ** -19, 19)
    i = -19
    max_lambda = 10 ** 19
    min_abs_error = float("inf")

    while max_lambda >= lambdA:
        w = regularized_linear_regression(Xtrain, ytrain, lambdA)
        error = mean_absolute_error(w, Xval, yval)

        if error < min_abs_error:
            min_abs_error = error
            bestlambda = lambdA
        lambdA *= 10
        i += 1
        lambdA = round(lambdA, abs(min(0, i)))

    return bestlambda


###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################

    i = 1
    multiplied_X, mapped_X_matrix = X, X
    while i < power:
        multiplied_X = np.multiply(multiplied_X, X)
        mapped_X_matrix = np.append(mapped_X_matrix, multiplied_X, axis=1)
        i += 1
    return mapped_X_matrix
