# -------------------- Import Libraries --------------------
import pandas as pd

# -------------------- Initialization and Input --------------------
# Dictionaries to store the calculated probabilities
probability_features = {}
probability_target = {}

# Get user input for the dataset path and the target column
dataset_path = input("Enter the dataset path: ")
target_variable = input("Enter the name of the target variable: ")

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(dataset_path)

# -------------------- Probability Calculations --------------------
# Get the unique classes from the target variable
distinct_target_values = df[target_variable].unique()

# --- Calculate P(Target | Feature) for each feature value ---
for column in df.columns:
    # Skip the target variable itself
    if column == target_variable:
        continue

    probability_features[column] = {}
    distinct_values = df[column].unique()

    for value in distinct_values:
        probability_features[column][value] = {}
        for target_value in distinct_target_values:
            # Count rows where feature has 'value' AND target has 'target_value'
            numerator = len(df[(df[column] == value) & (df[target_variable] == target_value)])
            
            # Count rows where feature has 'value'
            denominator = len(df[df[column] == value])
            
            # Calculate conditional probability and handle division by zero
            if denominator > 0:
                probability_features[column][value][target_value] = numerator / denominator
            else:
                probability_features[column][value][target_value] = 0

# --- Calculate Prior Probability P(Target) for each target class ---
total_rows = len(df)
for target_value in distinct_target_values:
    count_target = len(df[df[target_variable] == target_value])
    probability_target[target_value] = count_target / total_rows


# -------------------- Sort and Print --------------------
# Don't change below this line
sort_features = {q: {k: v for k, v in sorted(a.items() if a != None else {}, key=lambda item: item[0])} for q, a in sorted(probability_features.items(), key=lambda item: item[0])}
sort_target = {k: v for k, v in sorted(probability_target.items(), key=lambda item: item[0])}

print("\n--- Conditional Probabilities: P(Target | Feature) ---")
print(sort_features)
print("\n--- Prior Probabilities: P(Target) ---")
print(sort_target)