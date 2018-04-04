import numpy as np
from sklearn import datasets


class LogisticRegression:

    def __init__(self, tol=1e-1):
        self.tol = tol

    def fit(self, X, y, logging=False):
        """

        :param X: Array of shape [n_samples, n_feats]
        :param y: Array of shape [n_samples, 1]
        :param logging:
        :return:
        """
        self.num_classes = len(np.unique(y))
        rng = 0

        if self.num_classes > 2:
            self._W = np.random.uniform(-rng, rng, size=(self.num_classes, X.shape[1], 1))
            self._b = np.random.uniform(-rng, rng, size=(self.num_classes,))

            for i in range(self.num_classes):
                y_cls = y.copy()
                y_cls += 1
                y_cls[y_cls != i + 1] = 0
                y_cls[y_cls == i + 1] = 1
                self._binary_fit(X, y_cls, i, logging)
        else:
            self._W = np.random.uniform(-rng, rng, size=(1, X.shape[1], 1))
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
        if logging:
            print('Fitting for class {}'.format(cls))

        while True:
            pred = self._binary_predict(X, cls=cls)

            # [1, n_samples] x [n_samples, n_feats] => [1, n_feats]
            gradient = np.transpose(pred - y).dot(X)
            r = np.multiply(pred, (1 - pred))
            # [n_samples, n_samples]
            r_matrix = np.diagflat(r)
            # [n_feats, n_samples] x [n_samples, n_samples] x [n_samples, n_feats] => [n_feats, n_feats]
            hessian = np.transpose(X).dot(r_matrix).dot(X)

            # tr([1, n_feats]x[n_feats, n_feats]) => [n_feats, 1]
            delta = np.transpose(gradient.dot(np.linalg.inv(hessian)))
            W_old = self._W[cls].copy()
            self._W[cls] = W_old - delta

            if np.linalg.norm(self._W[cls] - W_old) < self.tol * np.linalg.norm(self._W[cls]):
                loss = self.loss(y, pred)
                pred = self._binary_predict(X, cls=cls)
                pred = np.round(pred)
                acc = np.mean(np.equal(y, pred))
                print('Loss: {}'.format(loss))
                print('Final Accuracy: {}'.format(acc))
                break

            if logging:
                loss = self.loss(y, pred)
                pred = self._binary_predict(X, cls=cls)
                pred = np.round(pred)
                acc = np.mean(np.equal(y, pred))
                print('Loss: {}'.format(loss))
                print('Accuracy: {}'.format(acc))

    def loss(self, y, pred):
        return - ((np.transpose(y).dot(np.log(np.maximum(pred, 1e-7))))
                  + (1 - np.transpose(y)).dot(np.log(np.maximum(1 - pred, 1e-7))))

    def _binary_predict(self, X, cls):
        # [n_samples, n_feats] x [n_feats, 1] => [n_samples, 1]
        linear = X.dot(self._W[cls])
        pred = 1.0 / (1.0 + np.exp(-linear))
        return pred

    def predict_proba(self, X):
        if self.num_classes > 2:
            pred = np.zeros((X.shape[0], self.num_classes))
            for i in range(self.num_classes):
                pred_cls = self._binary_predict(X, i)
                pred[:, i] = pred_cls.reshape((-1,))

            pred = pred / np.broadcast_to(np.sum(pred, axis=1).reshape((-1, 1)),
                                          (pred.shape[0], self.num_classes))

            return pred
        else:
            return self._binary_predict(X, 0)

    def predict(self, X):
        if self.num_classes > 2:
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1).reshape((-1, 1))
        else:
            return np.round(self._binary_predict(X, 0))


if __name__ == "__main__":
    model = LogisticRegression()
    X, y = datasets.load_iris(return_X_y=True)
    y = y.reshape((-1, 1))

    model.fit(X, y, logging=True)

    pred = model.predict(X)
    acc = np.mean(np.equal(y, pred))
    print('Accuracy: {}'.format(acc))
