from collections import Counter


class KNN:
    '''
    A simple implementation of the K-Nearest Neighbors algorithm.
    This class allows you to fit a model to training data and make predictions on test data.
    '''
    def __init__(self, k: int):
        self.points = None
        self.labels = None
        self.k = k
    
    def distance(self, point_a: list, point_b: list) -> float:
        '''
        Calculate the Euclidean distance between two points.
        :param point_a: First point as a list of coordinates.
        :param point_b: Second point as a list of coordinates.
        '''
        sum_dist = 0
        
        for i in range(0, len(point_a)):
            sum_dist += (point_a[i] - point_b[i]) ** 2
            dist = sum_dist ** 0.5
        return float(dist)
    
    def fit(self, X_train: list, y_train: list) -> None:
        '''
        Fit the KNN model to the training data.
        :param X_train: Training data as a list of points (each point is a list of coordinates).
        :param y_train: Labels for the training data as a list.
        '''
        self.points = X_train
        self.labels = y_train

    def predict(self, X_test: list) -> list:
        '''
        Predict the labels for the test data using the fitted KNN model.
        :param X_test: Test data as a list of points (each point is a list of coordinates).
        :return: List of predicted labels for the test data.
        '''
        predictions = []

        for test_point in X_test:
            distances = []

            for i in range(len(self.points)):
                train_point = self.points[i]
                train_label = self.labels[i]
                
                current_distance = self.distance(test_point, train_point)
                distances.append((current_distance, train_label))

            distances.sort(key=lambda x: x[0])
            k_nearest_neighbors = distances[:self.k]
            k_nearest_labels = [label for dist, label in k_nearest_neighbors]
            most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
            
            predictions.append(most_common_label)
        
        return predictions


if __name__ == "__main__":
    # Example usage
    X_train = [[1, 2], [2, 3], [3, 4], [5, 6]]
    y_train = [0, 0, 1, 1]
    X_test = [[1, 2], [2, 2], [3, 3], [4, 5]]

    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    
    print(predictions)  # Output: [0, 0, 0, 1]