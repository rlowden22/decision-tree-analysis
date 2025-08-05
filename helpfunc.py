
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Function to load and preprocess custom data
def load_and_preprocess_custom_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop('class', axis=1)
    y = df['class']
    
    # Encoding categorical columns (one-hot encoding)
    X = pd.get_dummies(X)
    return X, y

# Optional: Visualization function for Decision Tree (if desired later)
def plot_decision_tree(model):
    from sklearn.tree import plot_tree
    plt.figure(figsize=(20, 10))
    plot_tree(model, filled=True)
    plt.show()
