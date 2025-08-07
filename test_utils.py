import unittest
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from model_utils import load_data, split_data, train_decision_tree, evaluate_model, time_model_training

class TestDecisionTreeUtils(unittest.TestCase):

    """Set up the test class with a small subset of mushroom data."""
    @classmethod
    def setUpClass(cls):
        # Use a small subset of mushroom data for fast testing
        cls.filepath = "data/mushrooms.csv"
        cls.X, cls.y = load_data(cls.filepath)
        cls.X_small = cls.X[:100] 
        cls.y_small = cls.y[:100]

    """Test loading data from a CSV file."""
    def test_split_data(self):
        X_train, X_test, y_train, y_test = split_data(self.X_small, self.y_small, test_size=0.2)
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)

    """Test splitting data into training and testing sets."""
    def test_train_decision_tree(self):
        X_train, X_test, y_train, y_test = split_data(self.X_small, self.y_small)
        model = train_decision_tree(X_train, y_train)
        self.assertIsInstance(model, DecisionTreeClassifier)
        self.assertTrue(hasattr(model, "predict"))

    """Test training a decision tree classifier."""
    def test_evaluate_model_accuracy(self):
        X_train, X_test, y_train, y_test = split_data(self.X_small, self.y_small)
        model = train_decision_tree(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)
        self.assertGreaterEqual(accuracy, 0.5) 

    """Test evaluating the model's accuracy."""
    def test_time_model_training_output(self):
        sample_sizes = [10, 50]
        results = time_model_training(self.X_small, self.y_small, sample_sizes)
        self.assertEqual(len(results), 2)
        for size, acc, dur in results:
            self.assertIn(size, sample_sizes)
            self.assertIsInstance(acc, float)
            self.assertIsInstance(dur, float)

if __name__ == "__main__":
    unittest.main()
