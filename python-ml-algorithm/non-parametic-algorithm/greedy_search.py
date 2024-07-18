import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def greedy_search(features: np.ndarray, target: np.ndarray, max_features: int = 3):
    num_features = features.shape[1]
    best_features = []
    best_score = 0
    
    while len(best_features) < max_features:
        scores = []
        for i in range(num_features):
            if i not in best_features:
                current_features = best_features + [i]
                X_train, X_test, y_train, y_test = train_test_split(features[:, current_features], target, test_size=0.2, random_state=0)
                
                model = LogisticRegression(max_iter=200)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                scores.append((score, i))
        
        scores.sort(reverse=True)
        best_new_score, best_new_feature = scores[0]
        
        if best_new_score > best_score:
            best_score = best_new_score
            best_features.append(best_new_feature)
        else:
            break
            
    return best_features



