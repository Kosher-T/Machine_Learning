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
df = pd.read_csv("C:\\Code\\Code\\.Files\\magic04.data", names=cols)

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
    X = scaler.fit_transform(X)
    
    # Optionally balance classes using random oversampling
    if oversample:
        rods = RandomOverSampler()
        X, y = rods.fit_resample(X, y)
    
    # Combine features and target for convenience
    data = np.hstack((X, np.reshape(y, (-1, 1))))

    return data, X, y   # Return the processed data