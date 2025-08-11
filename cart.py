# Rowan Lowden
# 5008: Summer 2025
# Final Project Code: Decision Tree implmentation on 3 datasets
# This code implements an original CART Decision Tree model on various datasets.

import numpy as np
from collections import Counter


""" Defines the Node class for the Decision Tree. Either a decision node or a leaf node."""
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature          # index of feature to split on
        self.threshold = threshold      # value of the split
        self.left = left                # left subtree/child
        self.right = right              # right subtree/child
        self.value = value              # class label if it's a leaf
        
    # If value is not None and has no children, this is a leaf node
    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeRowan:
    """ initial values for the Decision Tree parameters """
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None # Root node of the tree, empty initially

    """ Fits the Decision Tree model to the training data. The tree learns from the data. """
    def fit(self, X, y): # X is the feature matrix (NumPy), y is the labels
        self.root = self._build_tree(X, y) # Build the tree recursively and stores it in self.root

    def _build_tree(self, X, y, depth=0): 
        num_samples, num_features = X.shape # Get the number of samples and features
        num_labels = len(np.unique(y)) # countsof unique labels in dataset

        # stopping criteria to make a leaf node
        # If max depth is reached, or if all samples belong to the same class, etc
        if (depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find the best feature and threshold to split on based on GIni impurity
        best_feat, best_thresh = self._best_split(X, y)
        if best_feat is None:
            return Node(value=self._most_common_label(y))

        # Create a decision node with the best feature and threshold
        left_idxs = X[:, best_feat] < best_thresh
        right_idxs = ~left_idxs # Get the indices for left and right splits

        # Recursively build the left and right subtrees
        left = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        # Return a new Node with the best feature and threshold, and the left/right subtrees
        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    """ Finds the best feature and threshold to split the data based on Gini impurity. """
    def _best_split(self, X, y):
        best_gain = -1 # Initialize best gain to a low value so any will be better
        best_feat, best_thresh = None, None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            #test each threshold for the current feature
            for threshold in thresholds:
                # Create left and right splits based on the threshold
                left_idxs = X[:, feature_index] < threshold
                right_idxs = ~left_idxs

                # If either split is empty, skip this threshold
                if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
                    continue

                # Calculate the Gini gain from this split
                gain = self._gini_gain(y, y[left_idxs], y[right_idxs])

                # If this gain is better than the best found so far, update best values
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feature_index
                    best_thresh = threshold

        return best_feat, best_thresh

    """ Calculates the Gini impurity for a set of labels (y). """
    def _gini(self, y):
        counts = np.bincount(y) # Count occurrences of each class label
        probs = counts / len(y) # Calculate probabilities of each class
        return 1 - np.sum(probs ** 2) #gini formula

    """ Calculates the Gini gain from a split. How much the Gini impurity is reduced by splitting the data from parent to child. """
    def _gini_gain(self, parent, left, right):
        weight_left = len(left) / len(parent) #proportion of samples in left split
        weight_right = len(right) / len(parent) #proportion of samples in right split
        return self._gini(parent) - (weight_left * self._gini(left) + weight_right * self._gini(right))

    """ Returns the most common label in the dataset. Used to create leaf nodes. """
    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0] #python counter builtin to get most common label

    """ Predicts the class labels for a set of samples. Uses the trained tree to make predictions. """
    def predict(self, X): # X is a 2D NumPy array of samples, collects predicitions to compare with true labels
        return np.array([self._traverse_tree(x, self.root) for x in X])

    """ Traverses the tree to make a prediction for a single sample for the predict fucntion. """
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():  # If it's a leaf node, return the class 
            return node.value
        if x[node.feature] < node.threshold: # if the feature value is less than the threshold, go left
            return self._traverse_tree(x, node.left)
        else: # if the feature value is greater than or equal to the threshold, go right
            return self._traverse_tree(x, node.right)