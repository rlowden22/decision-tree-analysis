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

# Load the dataset
df = pd.read_csv("data/mushrooms.csv")

# Split features and labels
X = df.drop("class", axis=1)
y = df["class"]

# Convert categorical features using one-hot encoding
X = pd.get_dummies(X)

# Convert to numpy arrays
X = X.to_numpy()
y = pd.factorize(y)[0]  # Convert 'edible'/'poisonous' to 0/1

# Define sample sizes for runtime testing
sample_sizes = [500, 1000, 2000, 5000, 8000]
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