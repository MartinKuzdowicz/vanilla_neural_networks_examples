import numpy as np


def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


x = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

np.random.seed(1)

# synapses
syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

# training setps
for i in xrange(6000):
    # layers
    l0 = x  # first layer is input data
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))

    # error is calculated on last layer
    # calculate error
    l2_error = y - l2

    if (i % 10000) == 0:
        print 'Error: {}'.format(str(np.mean(np.abs(l2_error))))

    # backpropagation
    l2_delta = l2_error * sigmoid(l2, deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * sigmoid(l1, deriv=True)

    # update weights (gradient descent)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print 'Output after training: {}'.format(l2)