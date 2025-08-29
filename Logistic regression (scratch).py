# Import necessary libraries
import numpy as np                           # For numerical operations and array handling
import pandas as pd                          # For data manipulation and analysis
import matplotlib.pyplot as plt              # For data visualization (plotting histograms)
from sklearn.preprocessing import StandardScaler   # For feature scaling (standardization)
from imblearn.over_sampling import RandomOverSampler   # For handling class imbalance by oversampling

# Define column names for the dataset
cols = [
    "fLength", "fWidth", "fSize", "fConc", "fConc1",
    "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"
]

# Load the MAGIC gamma telescope dataset into a pandas DataFrame
# The data is expected to be in the specified path with columns named as above
df = pd.read_csv("C:\\Code\\Code\\Data and files\\magic04.data", names=cols)

# Convert the class column to binary: 1 if 'g' (gamma), 0 otherwise (hadron)
df["class"] = (df["class"] == 'g').astype(int)

# Plot histograms for each feature, separated by class (gamma vs hadron)
for label in cols[:-1]:  # Exclude the 'class' column itself
    plt.hist(
        df[df["class"] == 1][label],
        color='blue',
        label='gamma',
        alpha=0.7,
        density=True
    )
    plt.hist(
        df[df["class"] == 0][label],
        color='red',
        label='hadron',
        alpha=0.7,
        density=True
    )
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()

# Randomly shuffle and split the dataset into train (60%), validation (20%), and test (20%) sets
train, valid, test = np.split(
    df.sample(frac=1),  # Shuffle the dataset
    [int(0.6*len(df)), int(0.8*len(df))]
)

def scale_dataset(dataframe, oversample=False):
    """
    Standardizes features and optionally applies random oversampling for class balance.
    
    Parameters:
        dataframe (DataFrame): Input data including features and class label.
        oversample (bool): Whether to balance classes using random oversampling.
        
    Returns:
        data (ndarray): Combined scaled features and labels.
        X (ndarray): Scaled feature matrix.
        y (ndarray): Target vector.
    """
    # Separate features and target
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values
    
    # Standardize features to zero mean and unit variance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # fit gets the mean and std, transform applies the formula (value - mean) / std to each feature
    
    # Optionally balance classes using random oversampling
    if oversample:
        rods = RandomOverSampler()
        X_res, y_res = rods.fit_resample(X, y)  # fit finds the value in (y) that appears the least number of times, resample duplicates random
                                     # rows of the least appearing value until the len matches the most appearing value, making them equal

    # Combine features and target for convenience
    data = np.hstack((X, np.reshape(y, (-1, 1))))  # np.hstack() takes one argument, a tuple of arrays to stack horizontally
                                                   # np.reshaped turns our horizontal y into a vertical column vector to match the shape of X
                                                   # Combined, they simply create an array where y is just another column at the end of X

    return data, X, y   # Return the processed data


train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)





# -------------------------
# Logistic Regression (from Scratch)
# -------------------------

def sigmoid(z):
    """Squash values into range (0, 1)."""
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.1, epochs=1000):
    """
    Trains logistic regression using gradient descent.
    
    Args:
        X (ndarray): Feature matrix (num_samples x num_features)
        y (ndarray): Labels (0 or 1)
        lr (float): Learning rate
        epochs (int): Training iterations
    
    Returns:
        W (ndarray): Learned weights
        b (float): Learned bias
    """
    m, n = X.shape
    W = np.zeros(n)  # initialize weights
    b = 0            # initialize bias

    for _ in range(epochs):
        # 1. Linear combination
        z = np.dot(X, W) + b
        # 2. Prediction with sigmoid
        y_hat = sigmoid(z)
        # 3. Gradients
        dw = (1/m) * np.dot(X.T, (y_hat - y))
        db = (1/m) * np.sum(y_hat - y)
        # 4. Update step
        W -= lr * dw
        b -= lr * db
    
    return W, b

def predict(X, W, b):
    """Predicts binary labels (0 or 1) from input features."""
    probs = sigmoid(np.dot(X, W) + b)
    return (probs >= 0.5).astype(int)

# -------------------------
# Train & Evaluate
# -------------------------
W, b = train_logistic_regression(X_train, y_train, lr=0.1, epochs=5000)

train_preds = predict(X_train, W, b)
valid_preds = predict(X_valid, W, b)
test_preds  = predict(X_test,  W, b)

print(f"Train Accuracy: {np.mean(train_preds == y_train) * 100:.2f}%")
print(f"Valid Accuracy: {np.mean(valid_preds == y_valid) * 100:.2f}%")
print(f"Test Accuracy : {np.mean(test_preds == y_test) * 100:.2f}%")

