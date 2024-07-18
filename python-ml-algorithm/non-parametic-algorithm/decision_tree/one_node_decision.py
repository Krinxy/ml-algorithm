import numpy as np

def decision_tree(example: np.ndarray, featuressplit: int = 0 , split_threshold:float = 0.0, largerthan: bool = True):
    feature = example[featuressplit]
    if largerthan:
        if feature > split_threshold:
            return 1
        return 0
    else:
        if feature <= split_threshold:
            return 1
        return 0
