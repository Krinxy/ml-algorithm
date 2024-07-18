import numpy as np


def accuracy_of_classification(labels: np.ndarray, results: np.ndarray):
    labels = labels.ravel().astype(int)
    results = results.ravel().astype(int)
    assert (labels.shape[0] == results.shape[0])
    correct = 0.0
    for i in range(labels.shape[0]):
        if labels[i] == results[i]:
            correct += 1
    correct = correct / labels.shape[0]
    return correct
