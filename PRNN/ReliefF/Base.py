import numpy as np


class Base(object):

    def fit(self, X, y):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X, y):
        raise NotImplementedError

    def _check_array(self, data):
        if isinstance(data, list):
            data = np.asarray(data, dtype="O")
        assert isinstance(data, np.ndarray), "input should be an array!"
        return data