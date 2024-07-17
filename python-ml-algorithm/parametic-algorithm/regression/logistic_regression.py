from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def Regression(x_train: np.ndarray, y_train: np.ndarray):
    model = LogisticRegression().fit(x_train, y_train)
    return model


def plot_logistic_curve(X, y, model):
    X_test = np.linspace(np.min(X), np.max(X), 300).reshape(-1, 1)      # Predict + Truth --> Class 1
    probabilities = model.predict_proba(X_test)[:, 1]

    # Plot der logistischen Kurve
    plt.figure(figsize=(10, 6))
    plt.scatter(X.ravel(), y, color='black', zorder=20, marker='o', label='Datenpunkte')
    plt.plot(X_test, probabilities, color='red', linewidth=3, label='Logistische Kurve')
    plt.xlabel('Merkmal')
    plt.ylabel('Wahrscheinlichkeit')
    plt.title('Logistische Regression: Logistische Kurve')
    plt.legend(loc='best')
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    data = np.random.randn(1000, 1)                 # 1000 Datenpunkte, 1 Merkmal
    labels = (np.random.rand(1000)> 0.5).astype(int)     # Binäre Labels (0 oder 1)

    data_train, data_test, label_train, label_test = train_test_split(data,
                                                                      labels, test_size=0.2, random_state=42)
    regression = Regression(data_train, label_train)
    predictions = regression.predict(data_test)

    accuracy = accuracy_score(label_test, predictions)
    print('Accuracy: ', accuracy)

    # TODO: "Logistic Curve"... STRAIGHT LINE. CHANGE. IT.

    plot_logistic_curve(data, labels, regression)
