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

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt 

def load_data(filepath):
    df = pd.read_csv(filepath)

    if 'class' in df.columns:  # Mushroom dataset
        X = df.drop('class', axis=1)
        y = df['class']
        X = pd.get_dummies(X)  # all features are categorical

    elif 'Species' in df.columns:  # Iris dataset
        X = df.drop('Species', axis=1)
        y = pd.factorize(df['Species'])[0]  # Convert species to integers

    elif 'diabetes' in df.columns:  # Diabetes dataset 
        X = df.drop('diabetes', axis=1)
        y = df['diabetes']  # Already 0 or 1
        X = pd.get_dummies(X)

    else:
        raise ValueError("Unrecognized dataset format.")

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
    
def save_results_to_csv(results, dataset_name, folder="results"):
    """
    Save empirical analysis results to a CSV file.
    
    Args:
        results (list of tuples): Each tuple should contain (Sample Size, Accuracy, Time)
        dataset_name (str): Name of the dataset (e.g., 'mushroom', 'diabetes', 'iris')
        folder (str): Folder to save the CSV files
    """
    # Create filename like: results/diabetes_accuracy_timing.csv
    filename = f"{dataset_name.lower()}_accuracy_timing.csv"
    filepath = os.path.join(folder, filename)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Write the CSV file
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Sample Size", "Accuracy", "Time (s)"])
        writer.writerows(results)
