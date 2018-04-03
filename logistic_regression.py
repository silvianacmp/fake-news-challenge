import numpy as np
from sklearn import datasets


class LogisticRegression:

    def __init__(self, learning_rate=1e-4, tol=1e-5):
        self.learning_rate = learning_rate
        self.tol = tol

    def fit(self, X, y, logging=False):
        """

        :param X: Array of shape [n_features, n_samples]
        :param y: Array of shape [1, n_samples]
        :param logging:
        :return:
        """
        self.num_classes = len(np.unique(y))
        rng = 1 / X.shape[1]

        if self.num_classes > 2:
            self._W = np.random.uniform(-rng, rng, size=(self.num_classes, 1, X.shape[0]))
            self._b = np.random.uniform(-rng, rng, size=(self.num_classes,))

            for i in range(self.num_classes):
                y_cls = y.copy()
                y_cls[y_cls != i] = 0
                self._binary_fit(X, y_cls, i, logging)
        else:
            self._W = np.random.uniform(-rng, rng, size=(1, 1, X.shape[0]))
            self._b = np.random.uniform(-rng, rng, size=(1,))
            self._binary_fit(X, y, 0, logging)

    def _binary_fit(self, X, y, cls, logging=False):
        """

        :param cls: the class for which the prediction is being made.
        :param logging: print loss at each step
        :param X: Array of shape [n_features, n_samples]
        :param y: Array of shape [1, n_samples]
        :return:
        """
        n_samples = X.shape[1]

        prev_loss = np.inf
        pred = self._binary_predict(X, cls)
        loss = self.loss(y, pred)

        if logging:
            print('Initial loss: {}'.format(loss))

        while prev_loss - loss > self.tol:
            prev_loss = loss

            # [n_samples, 1]
            self._W[cls] -= self.learning_rate * (1.0 / n_samples) * (pred - y).dot(X.transpose())
            self._b[cls] -= self.learning_rate * (1.0 / n_samples) * np.sum(pred - y)

            # [n_samples, n_features] x [n_features, 1] => [n_samples, 1]
            pred = self._binary_predict(X, cls)
            loss = self.loss(y, pred)

            if logging:
                print('Loss: {}'.format(loss))

            pred = model.predict(X)
            pred = np.round(pred)
            acc = np.mean(np.equal(y, pred))
            print('Accuracy: {}'.format(acc))

    def loss(self, y, pred):
        return -np.mean(np.multiply(y, np.log(pred)) + np.multiply((1.0 - y), np.log(1.0 - pred)))

    def _binary_predict(self, X, cls):
        linear = self._W[cls].dot(X) + self._b[cls]
        pred = 1.0 / (1.0 + np.exp(-linear))
        return pred

    def predict(self, X):
        if self.num_classes > 2:
            pred = np.zeros((X.shape[0], self.num_classes))
            for i in range(self.num_classes):
                pred_cls = self._binary_predict(X, i)
                pred[i] = pred_cls

            pred = pred / np.sum(pred, axis=1)
            return pred
        else:
            return self._binary_predict(X, 0)

if __name__ == "__main__":
    model = LogisticRegression(learning_rate=0.00001)

    X = np.zeros((2, 10))
    y = np.zeros((1, 10))
    for i in range(5):
        X[0, i] = i
        X[1, i] = i
        y[0, i] = 0

    for i in range(5, 10):
        X[0, i] = i
        X[1, i] = i
        y[0, i] = 1

    model.fit(X, y,logging=True)

    pred = model.predict(X)
    pred = np.round(pred)
    acc = np.mean(np.equal(y, pred))
    print('Accuracy: {}'.format(acc))
