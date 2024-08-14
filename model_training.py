from sklearn.model_selection import cross_val_predict, GridSearchCV
from utils import save_best_params_to_file

def train_sklearn_model(model, X, y, params=None, cv=5):
    """
    Trains a given sklearn model using cross-validation, performs hyperparameter tuning (if params are provided),
    and returns predictions, probabilities, and the trained model.

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

    # Predict values using cross-validation
    y_pred = cross_val_predict(model, X, y, cv=cv, method='predict')
    y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]

    # Return the model and predictions
    return model, y_pred, y_pred_proba, model.__class__.__name__
