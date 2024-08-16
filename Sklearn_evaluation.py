import os
import mlflow
import pickle
from Sklearn_training import train_sklearn_model
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from utils import (
    evaluate_and_save_results,
    load_model_configs,
    load_preprocessed_data,
    log_to_mlflow,
    calculate_metrics
)

def main():
    """
    Main function to load data, train models, evaluate them, log results to MLflow, and save models.
    """
    # Load preprocessed data
    X, y = load_preprocessed_data()

    # Load model configurations from JSON
    models_params = load_model_configs('Models/model_configs.json')

    # Available model classes
    model_classes = {
        "LogisticRegression": LogisticRegression,
        "XGBClassifier": XGBClassifier,
        "SVC": SVC
    }

    # Directory to save trained models
    saved_models_dir = 'Models/SavedModels'
    if not os.path.exists(saved_models_dir):
        os.makedirs(saved_models_dir)
        print(f"Directory {saved_models_dir} created.")
    else:
        print(f"Directory {saved_models_dir} already exists.")

    # Train and evaluate each model
    for model_name, mp in models_params.items():
        print(f"Training and evaluating model: {model_name}\n")
        model_class = model_classes.get(mp['model'])
        if model_class is None:
            raise ValueError(f"Model class {mp['model']} is not defined in model_classes.")

        model = model_class(random_state=42, probability=True) if 'SVC' in model_name else model_class(random_state=42)
        params = mp['params']

        # Create and set a new MLflow experiment for each model
        experiment_name = model_name
        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        # Train the model and get the best parameters
        best_model, y_pred, y_pred_proba, model_name = train_sklearn_model(model, X, y, params=params, cv=5)

        # Evaluate and save the results
        evaluate_and_save_results(model_name, y, y_pred, y_pred_proba)

        # Log results to MLflow
        metrics = calculate_metrics(y, y_pred, y_pred_proba)
        log_to_mlflow(best_model, metrics, run_name=model_name)

        # Save the trained model to a file in pickle format
        model_save_path = os.path.join(saved_models_dir, f"{model_name}.pkl")
        with open(model_save_path, 'wb') as model_file:
            pickle.dump(best_model, model_file)
        print(f"Model {model_name} saved to {model_save_path}")
        print("-" * 50)

if __name__ == "__main__":
    main()
