# Rowan Lowden
# 5008: Summer 2025
# Final Project Code: Decision Tree implmentation on mushroom data
#is the mushroom edible or poisnoius? 

#load the mushrooms
#split the data into training and test sets
#train decision tree model using sklearn
#evaluate the model's preformance 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the mushrooms dataset
def load_data():
    df = pd.read_csv('data/mushrooms.csv')  # Load the CSV file from the data folder
    X = df.drop('class', axis=1)  # Features (everything except the target column)
    y = df['class']  # Target (class: edible or poisonous)
    
    # Convert categorical variables into numeric (using one-hot encoding)
    X = pd.get_dummies(X)  # One-hot encode categorical columns
    return X, y

# Split the data into training and testing sets
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree model
def train_model(X_train, y_train):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

# Evaluate the model's performance
def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)  # Make predictions on the test set
    accuracy = accuracy_score(y_test, y_pred)  # Accuracy metric
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))  # Detailed classification report
    return accuracy

# Optional: Visualize the accuracy results (optional for later experimentation)
def visualize_results(accuracies):
    plt.plot(accuracies)
    plt.xlabel('Dataset Size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Dataset Size')
    plt.show()

if __name__ == '__main__':
    # Load data
    X, y = load_data()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)

    # Optionally visualize results
    visualize_results([accuracy])
