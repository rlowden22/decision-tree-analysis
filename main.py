# Rowan Lowden
# 5008: Summer 2025
# Final Project Code: Decision Tree implmentation on mushroom data
#is the mushroom edible or poisnoius? 


from model_utils import load_data, split_data, train_decision_tree, evaluate_model, save_results_to_csv, time_model_training #visualize_tree


def main():

    #picking data set
    filepath = 'data/diabetes_dataset.csv' #'data/mushrooms.csv'   # OR 'data/iris_100k.csv' #'data/diabetes_dataset.csv'

    # 1. Load the dataset
    X, y = load_data(filepath)

    # 2. Split the data into training and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Train the decision tree model
    model = train_decision_tree(X_train, y_train) 

    # 4. Evaluate the model
    evaluate_model(model, X_test, y_test)

    # 5. Empirical analysis: how model scales with data size
    #sample_sizes = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]

    sample_sizes = [500, 1000, 2000, 5000, 10000, 15000, 20000,
               30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    results = time_model_training(X, y, sample_sizes)

    print("\nEmpirical Results (Size | Accuracy | Time in seconds):")
    for size, acc, duration in results:
        print(f"{size}\t{acc:.4f}\t{duration:.4f} sec")

    # 6. Empirical analysis: how model scales with data size
    save_results_to_csv(results, dataset_name= "diabetes")
    

if __name__ == "__main__":
    main()