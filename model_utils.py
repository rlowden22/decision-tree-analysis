# Rowan Lowden
# 5008: Summer 2025
# Final Project Code: Decision Tree implmentation on mushroom data
# is the mushroom edible or poisnoius? 

#load the mushrooms
#split the data into training and test sets
#train decision tree model using sklearn
#evaluate the model's preformance 

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import csv
import os


def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df.drop('class', axis=1)
    y = df['class']
    X = pd.get_dummies(X)
    return X, y


def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)

def train_decision_tree(X_train, y_train):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return accuracy

def time_model_training(X, y, sample_sizes):
    results = []

    for size in sample_sizes: #loop through the sample sizes
        #sample of the data
        X_sample = X[:size]
        y_sample = y[:size]

        #split into the train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
        
        #train and time the model
        start = time.time()
        model = train_decision_tree(X_train, y_train)
        end = time.time()
        
        #evalulate the model with accuracy score
        accuracy = evaluate_model(model, X_test, y_test)

        #timing 
        duration = end - start
        #save results
        results.append((size, accuracy, duration))

        return results
    
def save_results_to_csv(results, filepath="results/accuracy_timing.csv"):
    """Save empirical analysis results to a CSV file."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Sample Size", "Accuracy", "Time (s)"])
        writer.writerows(results)