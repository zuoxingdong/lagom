import numpy as np


def rastrigin(x):
    # Search domain: [-5.12, 5.12]
    A = 10
    y = A*len(x)
    for x_part in x:
        y += x_part**2 - A*np.cos(2*np.pi*x_part)

    return y


def sphere(x):
    # Search domain: [-1000, 1000]
    y = 0.0
    for x_part in x:
        y += x_part**2

    return y


def holder_table(x):
    # Search domain: [-10, 10]
    x, y = x

    y = -np.abs(np.sin(x)*np.cos(y)*np.exp(np.abs(1 - np.sqrt(x**2 + y**2)/np.pi)))

    return y


def styblinski_tang(x):
    # Search domain: [-5, 5]
    y = 0.0
    for x_part in x:
        y += x_part**4 - 16*x_part**2 + 5*x_part

    return 0.5*y
