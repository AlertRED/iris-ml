from collections import Counter
from random import randint

import numpy as np


class Model():
    def __init__(self, X, Y):
        labels = list(set(Y))
        Y = list(map(lambda x: labels.index(x) + 1, Y))

        for i in range(len(Y)):
            j = randint(0, len(Y) - 1)
            X[i], X[j] = X[j], X[i]
            Y[i], Y[j] = Y[j], Y[i]

        self.X = np.array(X)
        self.Y = np.array(Y)

        np.random.seed(1)
        self.weights = 2 * np.random.random(len(X[0])) - 1

    def train(self, epoch=1, alpha=1.):

        for iteration in range(epoch):
            total_error = 0
            for row_index in range(len(self.Y)):
                x = self.X[row_index]
                y = self.Y[row_index]
                prediction = x.dot(self.weights)

                total_error += (y - prediction) ** 2

                delta = prediction - y
                self.weights -= alpha * (x * delta)
            yield {'error': (total_error / len(self.Y))}

    def test(self):
        c = Counter()
        for row_index in range(len(self.Y)):
            x = self.X[row_index]
            y = self.Y[row_index]
            prediction = x.dot(self.weights)
            c[round(prediction) == y] += 1
        return c.get(0, 0) / c.get(1, 1)


def get_dataset():
    X = []
    Y = []
    with open('iris.data', 'r') as file:
        for raw in file.read().split('\n'):
            if len(raw) != 0:
                lst = raw.replace('\n', '').split(',')
                X.append(list(map(float, lst[:-1])))
                Y.append(lst[-1])
    return X, Y


if __name__ == '__main__':
    X, Y = get_dataset()
    m = Model(X, Y)

    for message in m.train(100, 0.0001):
        print(message['error'])
    error = m.test()
    print(f'Final error: {error}')
