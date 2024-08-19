import os
import pickle
import mlflow
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from utils import (
    evaluate_and_save_results,
    load_model_configs,
    load_preprocessed_data,
    log_to_mlflow,
    calculate_metrics,
    save_best_params_to_file
)

def train_sklearn_model(model, X, y, params=None, cv=5):
    """
    Train a given sklearn model using cross-validation, perform hyperparameter tuning (if params are provided),
    and return predictions, probabilities, and the trained model.

    :param model: The classification model (e.g., LogisticRegression, RandomForestClassifier).
    :param X: Feature dataset.
    :param y: Target labels.
    :param params: Parameter grid for GridSearchCV (optional).
    :param cv: Number of cross-validation folds (default is 5).
    :return: Trained model, predictions, predicted probabilities, and model name.
    """
    if params:
        grid_search = GridSearchCV(model, param_grid=params, cv=cv, scoring='accuracy', verbose=1)
        grid_search.fit(X, y)
        model = grid_search.best_estimator_  # Update model to the best found by GridSearchCV
        best_params = grid_search.best_params_
        print(f"Best Parameters: {best_params}")

        # Save the best parameters to a file
        save_best_params_to_file(model.__class__.__name__, best_params)

        # Generate predictions and probabilities based on the best model
        y_pred = cross_val_predict(model, X, y, cv=cv, method='predict')
        y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    else:
        # Predict values using cross-validation if no hyperparameter tuning is performed
        y_pred = cross_val_predict(model, X, y, cv=cv, method='predict')
        y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]

    # Fit the model on the entire dataset
    model.fit(X, y)

    return model, y_pred, y_pred_proba, type(model).__name__, best_params

def main():
    """
    Main function to load data, train models, evaluate them, log results to MLflow, and save models.
    """
    X, y = load_preprocessed_data()

    models_params = load_model_configs('Models/model_configs.json')

    model_classes = {
        "LogisticRegression": LogisticRegression,
        "XGBClassifier": XGBClassifier,
        "SVC": SVC
    }

    saved_models_dir = 'Models/SavedModels'
    if not os.path.exists(saved_models_dir):
        os.makedirs(saved_models_dir)

    for model_name, mp in models_params.items():
        model_class = model_classes.get(mp['model'])
        model = model_class(random_state=42, probability=True) if 'SVC' in model_name else model_class(random_state=42)
        params = mp['params']

        # Create and set experiment for each model
        experiment_name = model_name
        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=model_name):
            # Train the model
            best_model, y_pred, y_pred_proba, model_name = train_sklearn_model(model, X, y, params=params, cv=5)

            # Evaluate and log the results
            evaluate_and_save_results(model_name, y, y_pred, y_pred_proba)
            metrics = calculate_metrics(y, y_pred, y_pred_proba)
            log_to_mlflow(best_model, model_name, metrics, run_name=model_name, params=params)

            # Save the trained model as .pkl for scikit-learn
            model_save_path = os.path.join(saved_models_dir, f"{model_name}.pkl")
            with open(model_save_path, 'wb') as model_file:
                pickle.dump(best_model, model_file)
            print(f"Model {model_name} saved to {model_save_path}")

if __name__ == "__main__":
    main()
