class SimpleModel:
    def __init__(self):
        self.mode = {}

    def fit(self, X, y):
        # Write your code here
        # I will use a list to calculate mode
        unique_x = list(set(X))
        # Get the corresponding Y for each unique X
        for x in unique_x:
            corresponding_y = [y[i] for i in range(0, len(X)) if X[i] == x]  # Get all Y values where X matches unique_x
            count_of_ones = corresponding_y.count(1)
            count_of_zeros = corresponding_y.count(0)
            
            # If the count of 1s is greater than or equal to the count of 0s, the mode is 1.
            # This also handles a tie, defaulting to 1. You could also default to 0.
            if count_of_ones > count_of_zeros:
                self.mode[x] = 1
            else:
                self.mode[x] = 0
        
    def predict(self, X):
        # Write your code here
        # for each value in X, return the mode from the model
        predictions = []
        for x in X:
            if x in self.mode:
                predictions.append(self.mode[x])
            else:
                predictions.append(None)
        return predictions


def train_and_predict(X_train, y_train, X_test):
    model = SimpleModel()
    # Fit the model with X_train and y_train
    model.fit(X_train, y_train)
    # Predict the labels of X_test
    model.predict(X_test)
    # Return the predicted labels
    return model.predict(X_test)


if __name__ == "__main__":
    # Example usage
    X_train = [1, 2, 3, 1, 2, 3, 1]
    y_train = [0, 1, 0, 0, 1, 0, 1]
    X_test = [1, 2, 3, 4]

    predictions = train_and_predict(X_train, y_train, X_test)
    print(predictions)  # Output: [0, 1, 0, None]