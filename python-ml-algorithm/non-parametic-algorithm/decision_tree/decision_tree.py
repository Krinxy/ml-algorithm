from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def DecisionTree(data:np.ndarray, labels):
    model = DecisionTreeClassifier()
    model.fit(data, labels)
    
    return model

if __name__ == '__main__':
    np.random.seed(0)
    data = np.random.rand(150, 5)               # 150 Data, 5 Features
    labels = np.random.randint(0, 2, 150)       # Label 0... Label 1...
    data_train, data_test, label_train, label_test = train_test_split(data,
                                                                      labels, test_size=0.2, random_state=42)

    model = DecisionTree(data_train, label_train)
    predictions = model.predict(data_test)
    accuracy = accuracy_score(label_test, predictions)

    # Accuracy Alt.
    count = 0
    checkliste = (predictions == label_test)
    for wert in checkliste:
        count += 1

    self_accuracy = sum(checkliste) / count


    print('Predictions: ', predictions)
    print('Actual Labels: ', label_test)
    print('Accuracy: ', accuracy)
    print('Self Accuracy: ', self_accuracy)
