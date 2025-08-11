#Rowan Lowden
# tree_diagram.py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load your dataset (e.g., diabetes)
df = pd.read_csv("data/mushrooms.csv")

# Separate features and target
X = df.drop("class", axis=1)   
y = df["class"]
X = pd.get_dummies(X)

# Split the dataset (optional but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Limit depth for readability
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Plot the tree
plt.figure(figsize=(16, 10))
plot_tree(model,
          filled=True,
          feature_names=X.columns,
          class_names=model.classes_.astype(str),
          rounded=True)
plt.title("Simplified Decision Tree (Max Depth = 5)")

# Save the plot for my report
plt.savefig("images/tree_diagram_depth5.png", dpi=300, bbox_inches='tight')
plt.show()