from sklearn.linear_model import LinearRegression
import numpy as np


def Regression(x_train: np.ndarray, y_train: np.ndarray):
    model = LinearRegression().fit(x_train, y_train)

    return model
