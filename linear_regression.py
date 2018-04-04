import numpy as np
from sklearn import datasets


class LinearRegression:

    def __init__(self, learning_rate=0.1, tol=1e-5):
        self.learning_rate = learning_rate
        self.tol = tol

    def fit(self, X, y, logging=False):
        """

        :param X: Array of shape [n_samples, n_features]
        :param y: Array of shape [n_samples, 1]
        :return:
        """
        X = X.transpose()
        y = y.transpose()

        self._W = np.random.randn(1, X.shape[0])
        self._b = 0
        n_samples = X.shape[1]

        prev_loss = np.inf
        pred = self._predict(X)
        loss = self.loss(y, pred)
        if logging:
            print('Initial loss: {}'.format(loss))

        while prev_loss - loss > self.tol:
            prev_loss = loss

            # [1, n_samples] x [n_samples, n_feats] => [1, n_feats]
            self._W -= self.learning_rate * (1.0 / n_samples) * (pred - y).dot(X.transpose())
            self._b -= self.learning_rate * (1.0 / n_samples) * np.sum(pred - y)
            # [n_samples, n_features] x [n_features, 1] => [n_samples, 1]
            pred = self._predict(X)
            loss = self.loss(y, pred)
            if logging:
                print('Loss: {}'.format(loss))

        X = X.transpose()
        y = y.transpose()

    def fit_closed_form(self, X, y):
        XX = X.transpose().dot(X)
        Xy = X.transpose().dot(y)
        self._W = np.linalg.inv(XX).dot(Xy).transpose()
        self._b = 0
        pred = self.predict(X)
        loss = self.loss(y, pred)
        print('Loss closed form: {}'.format(loss))

    def _predict(self, X):
        # [1, n_features] x [n_features, n_samples] => [1, n_samples]
        return self._W.dot(X) + self._b

    def predict(self, X):
        return self._predict(X.transpose()).transpose()

    def loss(self, y, pred):
        return np.mean((y - pred) ** 2) / 2


if __name__ == "__main__":
    X, y = datasets.load_diabetes(return_X_y=True)
    y = y.reshape((-1, 1))
    model = LinearRegression(learning_rate=0.5, tol=0.00001)
    model.fit(X, y, logging=True)
    model.fit_closed_form(X, y)
