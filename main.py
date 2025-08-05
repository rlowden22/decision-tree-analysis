# Rowan Lowden
# 5008: Summer 2025
# Final Project Code: Decision Tree implmentation on mushroom data
#is the mushroom edible or poisnoius? 


from model_utils import load_data, split_data, train_decision_tree, evaluate_model, save_results_to_csv, time_model_training

def main():
    # 1. Load the dataset
    X, y = load_data("data/mushrooms.csv")

    # 2. Split the data into training and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Train the decision tree model
    model = train_decision_tree(X_train, y_train)

    # 4. Evaluate the model
    evaluate_model(model, X_test, y_test)

    # 5. Empirical analysis: how model scales with data size
    sample_sizes = [500, 1000, 2000, 4000, 8000]
    results = time_model_training(X, y, sample_sizes)

    print("\nEmpirical Results (Size | Accuracy | Time in seconds):")
    for size, acc, duration in results:
        print(f"{size}\t{acc:.4f}\t{duration:.4f} sec")

    # 6. Empirical analysis: how model scales with data size
    save_results_to_csv(results)
    print("Results saved to results/accuracy_timing.csv")

if __name__ == "__main__":
    main()