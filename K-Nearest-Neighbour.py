from collections import Counter


class KNN:
    def __init__(self, k):
        self.points = None
        self.labels = None
        self.k = k
    
    def distance(self, point_a, point_b):
        sum_dist = 0
        
        for i in range(0, len(point_a)):
            sum_dist += (point_a[i] - point_b[i]) ** 2
            dist = sum_dist ** 0.5
        return float(dist)
    
    def fit(self, X_train, y_train):
        self.points = X_train
        self.labels = y_train

    def predict(self, X_test):
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