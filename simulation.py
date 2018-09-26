#!/usr/bin/env python

import numpy as np
import torch
from torch.distributions import normal
from time import time
#import matplotlib.pyplot as plt
#import pylab

def gen_data(num_samples, sigma):

    #X = np.random.uniform(low=0.0, high=1.0, size=(num_samples, 1))
    X = torch.rand(num_samples, 1)
    #Z = np.random.normal(loc=0.0, scale=signma, size=(num_samples,1))
    Z = normal.Normal(0.0, sigma).sample((num_samples,1))
    Y = np.cos(2*np.pi*X) + Z

    return (X, Y)


def compute_mse(coefficients, X, Y):

    predicted_Y = torch.zeros_like(Y)
    for exponent, coefficient in enumerate(coefficients):
        predicted_Y += coefficient * (X ** exponent)

    # note: outputs torch of zero dimensions (use .shape or .size())
    MSE = torch.mean(torch.pow(Y - predicted_Y, 2))
    return MSE


def compute_testing_mse(coefficients, sigma):
    (x_test, y_test) = gen_data(2000, sigma)
    testing_loss = compute_mse(coefficients, x_test, y_test)
    return testing_loss.item()

# TODO: add SGD, mini-batch SGD
def fit_data(x, y, degree_polynomial, sigma, debug=False):

    num_coefficients = degree_polynomial + 1
    coefficients = torch.randn(num_coefficients, requires_grad=True)

    learning_rate = 0.1
    iteration=0
    previous_loss=1000
    while True:

        # compute loss (forward pass)
        loss = compute_mse(coefficients, x, y)

        if debug:
            if iteration % 10000 == 0:
                print("iteration " + str(iteration) + ": " + str(loss.item()))

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        loss.backward()

        loss_diff = abs(previous_loss - loss)

        #if loss_diff < 0.00001:
        if loss_diff < 0.00000001:

            if debug:
                print("final loss: " + str(loss.item()))
                print("final coeffs: " + str(coefficients))

            break

        # Update weights using gradient descent. For this step we just want to mutate
        # the values of w1 and w2 in-place; we don't want to build up a computational
        # graph for the update steps, so we use the torch.no_grad() context manager
        # to prevent PyTorch from building a computational graph for the updates
        with torch.no_grad():
            #w1 -= learning_rate * w1.grad
            #w2 -= learning_rate * w2.grad
            #learning_rate = loss.item() / 10
            coefficients -= learning_rate * coefficients.grad


            # Manually zero the gradients after running the backward pass
            #w1.grad.zero_()
            #w2.grad.zero_()
            coefficients.grad.zero_()

        previous_loss = loss
        iteration += 1

    testing_loss = compute_testing_mse(coefficients, sigma)

    return (coefficients, loss.item(), testing_loss)


def experiment(num_samples, degree_polynomial, sigma, debug=False):

    if debug:
        ts = time()

    num_trials = 50

    mse_sum=0
    test_mse_sum=0
    coefficients_sum=torch.zeros(degree_polynomial+1)
    for trial_idx in range(num_trials):

        if debug:
            print("trial iteration: " + str(trial_idx))

        # generate data
        (x, y) = gen_data(num_samples, sigma)

        # fit coefficients
        (coefficients, mse, test_mse) = fit_data(x, y, degree_polynomial, sigma)

        mse_sum += mse
        test_mse_sum += test_mse
        coefficients_sum += coefficients

    mse_avg = mse_sum / num_trials
    test_mse_avg = test_mse_sum / num_trials
    coefficients_avg = coefficients_sum / num_trials

    bias_mse = compute_testing_mse(coefficients_avg, sigma)

    if debug:
        te = time()
        total_time = te - ts
        print('Experiment took %2.4f seconds.' % (te - ts))

    return (mse_avg, test_mse_avg, bias_mse)

(mse_avg, test_mse_avg, bias_mse) = experiment(30, 4, 0.1, True)

print("MSE avg: " + str(mse_avg))
print("test MSE avg: " + str(test_mse_avg))
print("bias MSE: " + str(bias_mse))

'''
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
degree_polynomial = 4
num_samples = 30
sigma = 0.1

# generate data
(x, y) = gen_data(num_samples, sigma)

# fit coefficients
(coefficients, mse, test_mse) = fit_data(x, y, degree_polynomial, sigma)

print("MSE: " + str(mse))
print("Test MSE: " + str(test_mse))

# set up plot
#plt.figure()
pylab.figure()

# plot random dataset
#plt.scatter(x.numpy(), y.numpy())
pylab.scatter(x.numpy(), y.numpy())

# compute polynomial over grid of X
polynomial_x = torch.arange(start=0.0, end=1.0, step=0.001)
predicted_Y = torch.zeros_like(polynomial_x)
for exponent, coefficient in enumerate(coefficients):
    predicted_Y += coefficient * (polynomial_x ** exponent)

# plot polynomial
#plt.plot(polynomial_x.detach().numpy(), predicted_Y.detach().numpy())
pylab.plot(polynomial_x.detach().numpy(), predicted_Y.detach().numpy())

# show plot
pylab.show()
'''