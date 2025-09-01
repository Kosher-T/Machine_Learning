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
        X, y = rods.fit_resample(X, y)  # fit finds the value in (y) that appears the least number of times, resample duplicates random          # type: ignore
                                     # rows of the least appearing value until the len matches the most appearing value, making them equal

    # Combine features and target for convenience
    data = np.hstack((X, np.reshape(y, (-1, 1))))  # np.hstack() takes one argument, a tuple of arrays to stack horizontally
                                                   # np.reshaped turns our horizontal y into a vertical column vector to match the shape of X
                                                   # Combined, they simply create an array where y is just another column at the end of X

    return data, X, y   # Return the processed data


train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)




from sklearn.metrics import classification_report
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary crossentropy')
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    plt.show()

def train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs):
    nn_model = Sequential([
        Dense(num_nodes, activation='relu', input_shape=(10,)),
        Dropout(dropout_prob),
        Dense(num_nodes, activation='relu'),
        Dropout(dropout_prob),
        Dense(1, activation='sigmoid')
    ])

    nn_model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy',
                    metrics=['accuracy'])
    history = nn_model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0
    )

    return nn_model, history

least_val_loss = float('inf')
least_loss_model = None
epochs=100
for num_nodes in [16, 32, 64]:
    for dropout_prob in[0, 0.2]:
        for lr in [0.01, 0.005, 0.001]:
            for batch_size in [32, 64, 128]:
                print(f"{num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}")
                model, history = train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs)
                plot_history(history)
                val_loss = model.evaluate(X_valid, y_valid)[0]
                if val_loss < least_val_loss:
                    least_val_loss = val_loss
                    least_loss_model = model

if least_loss_model is not None:
    y_pred = least_loss_model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int).reshape(-1,)
    print(classification_report(y_test, y_pred))
else:
    print("No model was trained successfully. Please check your data and hyperparameters.")