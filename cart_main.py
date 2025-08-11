# Rowan Lowden
# 5008: Summer 2025
# Final Project Code: Decision Tree implmentation on 3 datasets
# This code implements an original CART Decision Tree model on various datasets.

import pandas as pd
import numpy as np
import time
from cart import DecisionTreeRowan  # your custom implementation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(filepath):
    df = pd.read_csv(filepath)

    if 'class' in df.columns:  # Mushroom
        X = pd.get_dummies(df.drop('class', axis=1))
        y = pd.factorize(df['class'])[0]  # edible/poisonous -> 0/1

    elif 'Species' in df.columns:  # Iris
        X = df.drop('Species', axis=1)
        y = pd.factorize(df['Species'])[0]

    elif 'diabetes' in df.columns:  # Clinical diabetes
        X = pd.get_dummies(df.drop('diabetes', axis=1))
        y = df['diabetes'].to_numpy()
        if not np.issubdtype(y.dtype, np.number):
            y = pd.factorize(y)[0]
    else:
        raise ValueError("Unrecognized dataset format.")

    return X.to_numpy(), np.asarray(y)

if __name__ == "__main__":
    # Pick ONE file here:
    filepath = "data/mushrooms.csv"
    #filepath = "data/iris_100k.csv"
    # filepath = "data/diabetes_dataset.csv"

    X, y = load_data(filepath)
    n = len(y)
    print(f"Loaded: {filepath} | samples={n} | features={X.shape[1]}")

# Define sample sizes for runtime testing
sample_sizes = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]
# sample_sizes = [500, 1000, 2000, 5000, 10000, 15000, 20000,30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

results = []

# Loop through different sample sizes
for size in sample_sizes:
    print(f"\n--- Testing on {size} samples ---")
    X_sample = X[:size]
    y_sample = y[:size]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

    # Train the custom Decision Tree and time it
    start_time = time.time()
    clf = DecisionTreeRowan(max_depth=10)  # depth can be adjusted
    clf.fit(X_train, y_train)
    end_time = time.time()

    duration = end_time - start_time

    # Make predictions
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy using sklearn's accuracy_score

    print(f"Time taken: {duration:.4f} seconds")
    print(f"Accuracy: {accuracy:.4f}")

    results.append((size, duration, accuracy))

# Print final summary
print("\nSummary:")
print("Sample Size | Time | Accuracy")
for size, duration, accuracy in results:
    print(f"{size:<12} {duration:.4f}     {accuracy:.4f}")