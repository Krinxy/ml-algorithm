import numpy as np
from decision_tree import one_node_decision
from compute_accuracy import accuracy_of_classification


def optimize_mini_tree_one_feature(features: np.ndarray, labels: np.ndarray):
    assert (features.shape[0] == labels.shape[0])

    best_feature = -1
    best_threshold = -987654321.0
    best_accuracy = -1.0
    best_mode = True

    # Check all features ...
    for f in range(features.shape[1]):
        num_tests = 10
        feature_values = features[:, f]
        min_val = np.min(feature_values)
        max_val = np.max(feature_values)
        # TODO: implement the checks, find the best values

        # Iterate through num_tests thresholds
        for t in range(num_tests):
            threshold = min_val + t * (max_val - min_val) / (num_tests - 1)

            # Test with larger_equal = True
            predictions = np.zeros_like(labels)
            for i in range(features.shape[0]):
                decision = one_node_decision(features[i], split_on_feature=f, split_threshold=threshold,
                                             larger_equal=True)
                predictions[i] = decision
            accuracy = accuracy_of_classification(labels, predictions)

            if accuracy > best_accuracy:
                best_feature = f
                best_threshold = threshold
                best_accuracy = accuracy
                best_mode = True

            predictions = np.zeros_like(labels)
            for i in range(features.shape[0]):
                decision = one_node_decision(features[i], split_on_feature=f, split_threshold=threshold,
                                             larger_equal=False)
                predictions[i] = decision
            accuracy = accuracy_of_classification(labels, predictions)

            if accuracy > best_accuracy:
                best_feature = f
                best_threshold = threshold
                best_accuracy = accuracy
                best_mode = False

    return best_feature, best_threshold, best_mode
