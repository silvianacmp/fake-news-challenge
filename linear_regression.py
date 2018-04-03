import numpy as np
from sklearn import datasets


class LinearRegression:

    def __init__(self, learning_rate=0.1, tol=1e-10):
        self.learning_rate = learning_rate
        self.tol = tol

    def fit(self, X, y, logging=False):
        """

        :param X: Array of shape [n_features, n_samples]
        :param y: Array of shape [1, n_samples]
        :return:
        """
        # [1, n_features]
        self._W = np.random.randn(1, X.shape[0])
        self._b = 0
        n_samples = X.shape[1]

        prev_loss = np.inf
        pred = self.predict(X)
        loss = self.loss(y, pred)
        if logging:
            print('Initial loss: {}'.format(loss))

        while prev_loss - loss > self.tol:
            prev_loss = loss

            # [1, n_samples] x [n_samples, n_feats] => [1, n_feats]
            self._W -= self.learning_rate * (1.0 / n_samples) * (pred - y).dot(X.transpose())
            self._b -= self.learning_rate * (1.0 / n_samples) * np.sum(pred - y)
            # [n_samples, n_features] x [n_features, 1] => [n_samples, 1]
            pred = self.predict(X)
            loss = self.loss(y, pred)
            if logging:
                print('Loss: {}'.format(loss))

    def predict(self, X):
        # [1, n_features] x [n_features, n_samples] => [1, n_samples]
        return self._W.dot(X) + self._b

    def loss(self, y, pred):
        return np.mean((y - pred) ** 2)


if __name__ == "__main__":
    model = LinearRegression(learning_rate=0.8)

    X = np.zeros((2, 10))
    y = np.zeros((1, 10))
    for i in range(10):
        X[0, i] = i
        X[1, i] = i
        y[0, i] = i

    model.fit(X, y, logging=True)
