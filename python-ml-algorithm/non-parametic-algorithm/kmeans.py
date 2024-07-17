from cv2 import kmeans
import cv2
import numpy as np
import matplotlib.pyplot as plt

def kmeans_algorithm(data,num_cluster: int = 0):
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 100, 0.2)
    values, labels, centers = kmeans(data, num_cluster, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    return values, labels.flatten(), centers


def plottingkmeans(data: np.ndarray, labels: np.ndarray, center: np.ndarray):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis', alpha=0.5)
    plt.scatter(center[:, 0], center[:, 1], c='red', s=200, marker='X', label='Cluster Centers')
    plt.title('k-Means Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    data = np.random.rand(100, 2) * 100
    clusters = 5
    _, labels, centers = kmeans_algorithm(data, clusters)       # _ is just data
    print('Labels of data:')
    print(labels)
    print('Center of cluster:')
    print(type(centers))

    plottingkmeans(data, labels, centers)
