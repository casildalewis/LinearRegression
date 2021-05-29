import numpy as np
from matplotlib import pyplot as plt
import math

def get_dataset(filename):
    """
    arguments: 
        filename - a string representing the path to the csv file.

    returns:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    dataset = np.genfromtxt(filename, dtype=float, delimiter = ",", skip_header = 1)
    dataset = np.delete(dataset, 0, 1)

    return dataset


def print_stats(dataset, col):
    """
    arguments: 
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on. 
                  For example, 1 refers to density.

    returns:
        None
    """
    colArray = dataset[:, col]
    rowNum = dataset.shape[0]

    mean = 0.0
    for num in colArray:
        mean += num
    mean /= rowNum

    stddev = 0.0
    for num in colArray:
        stddev += (num - mean)**2
    stddev /= (rowNum - 1)
    stddev = math.sqrt(stddev)

    print(rowNum)
    print("{:.2f}" .format(mean))
    print("{:.2f}" .format(stddev))




def regression(dataset, cols, betas):
    """
    arguments: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    returns:
        mse of the regression model
    """
    mse = 0.0
    n = dataset.shape[0]
    numFeat = len(cols)

    for row in dataset:
        y = row[0]
        error = 0.0

        error += betas[0] - y

        for i in range (numFeat):
            error += (row[cols[i]] * betas[i+1])
        error **= 2

        mse += error
        

    mse /= n

    return mse


def gradient_descent(dataset, cols, betas):
    """
    arguments: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    returns:
        An 1D array of gradients
    """
    grads = []
    n = dataset.shape[0]
    numFeat = len(cols)

    sumRow = 0.0
    for row in dataset:
        y = row[0]
        error = 0.0
        error += betas[0] - y

        for i in range (numFeat):
            error += (row[cols[i]] * betas[i+1])
        
        sumRow += error
    sumRow *= 2
    sumRow /= n

    grads.append(sumRow)

    for num in cols:

        sumRow = 0.0
        for row in dataset:
            y = row[0]
            error = 0.0
            error += betas[0] - y

            for i in range (numFeat):
                error += (row[cols[i]] * betas[i+1])
            error *= row[num]

            sumRow += error
        sumRow *= 2
        sumRow /= n

        grads.append(sumRow)

    return grads


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    arguments: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    """
    grads = gradient_descent(dataset, cols, betas)

    for i in range (T):
        for j in range (len(betas)):
            betas[j] -= eta*grads[j]

        mse = regression(dataset, cols, betas)

        grads = gradient_descent(dataset, cols, betas)

        line = str(i + 1) + " " + str("{:0.2f}".format(mse))
        for beta in betas:
            line += " " + str("{:0.2f}".format(beta))

        print(line)


def compute_betas(dataset, cols):
    """
    arguments: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    returns:
        A tuple containing corresponding mse and several learned betas
    """
    x = dataset[:, cols]
    ones = np.ones((dataset.shape[0], 1))
    x = np.hstack((ones, x))

    xTrans = np.transpose(x)

    y = dataset[:, 0]

    betas = (np.linalg.inv(xTrans @ x) @ xTrans) @ y

    mse = regression(dataset, cols, betas)

    return (mse, *betas)


def predict(dataset, cols, features):
    """
    arguments: 
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    returns:
        The predicted body fat percentage value
    """
    (mse, *betas) = compute_betas(dataset, cols)

    features = np.insert(features, 0, 1)

    result = np.transpose(betas) @ features
    return result


def synthetic_datasets(betas, alphas, X, sigma):
    """
    arguments:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise

    returns:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """
    linCol1 = []
    quadCol1 = []

    for x in X:
        z = np.random.normal(0, sigma)
        y = betas[0] + (betas[1] * x) + z
        linCol1.append(y)

        z = np.random.normal(0, sigma)
        y = alphas[0] + (alphas[1] * x * x) + z
        quadCol1.append(y)        

    lin = np.c_[linCol1, X]
    quad = np.c_[quadCol1, X]

    return (lin, quad)


def plot_mse():
    """
    Plot an MSE-sigma graph
    
    """
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    # TODO: Generate datasets and plot an MSE-sigma graph
    X = np.random.randint(low = -100, high = 101, size = (1000,1))

    betas = np.random.uniform(0.0, 1.0, size = 2)
    alphas = np.random.uniform(0.0, 1.0, size = 2)

    sigmas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

    linarray = []
    quadarray = []
    for sigma in sigmas:
        (lin, quad) = synthetic_datasets(betas, alphas, X, sigma)

        linmse, *linbetas = compute_betas(lin, [1])
        quadmse, *quadbetas = compute_betas(quad, [1])

        linarray.append(linmse)
        quadarray.append(quadmse)

    plt.plot(sigmas, linarray, label = "linear data", marker = "o")
    plt.plot(sigmas, quadarray, label = "quadratic data", marker = "o")
    plt.xlabel("sigma")
    plt.ylabel("MSE")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig("mse.pdf")


if __name__ == '__main__':
    plot_mse()

# def main():

#     dataset = get_dataset('bodyfat.csv')

#     #print_stats(dataset, 1)

#     # print(regression(dataset, cols=[2,3], betas=[0,0,0]))
#     # print(regression(dataset, cols=[2,3,4], betas=[0,-1.1,-.2,3]))

#     # dataset = dataset[0:5, 0:5]
#     # print(dataset)
#     # print(gradient_descent(dataset, cols=[2,3], betas=[0,0,0]))

#     # iterate_gradient(dataset, cols=[1,8], betas=[400,-400,300], T=10, eta=1e-4)

#     # print(compute_betas(dataset, cols=[1,2]))

#     # print(predict(dataset, cols=[1,2], features=[1.0708, 23]))

#     # (lin, quad) = synthetic_datasets(np.array([0,2]), np.array([0,1]), np.array([[4]]), 1)
#     # print(lin.shape, quad.shape)

# if __name__=="__main__": 
#     main()