import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# cost function
def compute_cost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


# gradient descent algorithm
def gradient_descent(X, y, theta, alpha, conv):
    cost = []
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    curr_error = (X * theta.T) - y
    prev_error = curr_error + np.ones(curr_error.shape) * (conv + 1)

    while abs(curr_error - prev_error).mean() > conv:
        prev_error = curr_error

        for j in range(parameters):
            term = np.multiply(curr_error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        cost.append(compute_cost(X, y, theta))
        theta = temp
        curr_error = (X * theta.T) - y

    return theta, cost


def main():
    # Get data
    path = os.getcwd() + "/data/ex1data1.txt"
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
    # print(data.head())
    # print(data.describe())
    # data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
    # plt.show()

    # feature normalization and addition of x_0 = 1s
    data = (data - data.mean()) / data.std()
    data.insert(0, 'Ones', 1)

    # set X (training data) and y (target variable)
    cols = data.shape[1]
    X = data.iloc[:, 0:cols - 1]
    y = data.iloc[:, cols - 1:cols]

    # print(X.head())
    # print(y.head())

    # convert X and y to matrices
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0, 0]))

    # set algorithm paramenters and calculate model parameters with GD
    alpha = 0.01  # step size for gradient descent
    conv = 10**(-15)  # convergence limit for error
    g, cost = gradient_descent(X, y, theta, alpha, conv)
    x = np.linspace(data.Population.min(), data.Population.max(), 100)
    f = g[0, 0] + (g[0, 1] * x)

    # plot linear approximation and corresponding data data
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.Population, data.Profit, label='Training Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')

    # plot value of cost function over iterations of GD algorithm
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(np.arange(len(cost)), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()


if __name__ == '__main__':
    main()
