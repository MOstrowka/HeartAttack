import os
import json
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    roc_curve, auc


def load_preprocessed_data():
    """
    Load preprocessed data from CSV files.

    :return: Tuple of features (X) and labels (y).
    """
    X = pd.read_csv('Data/X_preprocessed.csv').values
    y = pd.read_csv('Data/y_preprocessed.csv').values.ravel()
    return X, y


def load_model_configs(config_file):
    """
    Load model configurations from a JSON file.

    :param config_file: Path to the JSON file.
    :return: Dictionary containing model configurations.
    """
    with open(config_file, 'r') as file:
        model_configs = json.load(file)
    return model_configs


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate classification metrics.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param y_pred_proba: Predicted probabilities.
    :return: Dictionary of calculated metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }
    return metrics


def log_to_mlflow(model, metrics, run_name="Model", params=None):
    """
    Log model, metrics, and parameters to MLflow.

    :param model: Trained model object.
    :param metrics: Dictionary of calculated metrics.
    :param run_name: Name of the MLflow run.
    :param params: Dictionary of model parameters.
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics(metrics)
        if params:
            mlflow.log_params(params)

        # Log the model
        if hasattr(model, 'save'):
            mlflow.keras.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")


def save_best_params_to_file(model_name, best_params):
    """
    Save best model parameters to a JSON file.

    :param model_name: Name of the model.
    :param best_params: Dictionary of best parameters.
    """
    best_params_dir = 'Models/BestParams'
    os.makedirs(best_params_dir, exist_ok=True)
    file_path = os.path.join(best_params_dir, f'{model_name}_best_params.json')
    with open(file_path, 'w') as file:
        json.dump(best_params, file)
    print(f"Best parameters of model {model_name} saved to {file_path}")


def evaluate_and_save_results(model_name, y_true, y_pred, y_pred_proba):
    """
    Evaluate the model and save the results (metrics, confusion matrix, ROC curve) to a single Excel sheet.
    The results will be appended as new rows to the existing sheet.

    :param model_name: Name of the model.
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param y_pred_proba: Predicted probabilities.
    """
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    metrics['model'] = model_name  # Add model name to the metrics

    # Save metrics to a single Excel sheet
    results_dir = 'Results'
    os.makedirs(results_dir, exist_ok=True)
    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_file = os.path.join(results_dir, 'metrics.xlsx')

    if os.path.exists(metrics_file):
        # Load existing metrics and append the new ones
        existing_metrics_df = pd.read_excel(metrics_file)
        combined_df = pd.concat([existing_metrics_df, metrics_df], ignore_index=True)
    else:
        combined_df = metrics_df

    with pd.ExcelWriter(metrics_file, engine='openpyxl', mode='w') as writer:
        combined_df.to_excel(writer, sheet_name='Metrics', index=False)

    print(f"Metrics of model {model_name} saved to {metrics_file}")

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(results_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()

    # Save ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, f'{model_name}_roc_curve.png'))
    plt.close()

    print(f"Confusion matrix and ROC curve of model {model_name} saved in {results_dir}")