import os
import json
import mlflow
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    roc_curve, auc
from mlflow.models import infer_signature

def load_preprocessed_data():
    """
    Load preprocessed data from CSV files.

    :return: Tuple of features (X) and labels (y).
    """
    X = pd.read_csv('Data/X_preprocessed.csv')
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


def log_to_mlflow(model, model_name, metrics, run_name="Model", params=None):
    """
    Logowanie modelu, metryk i parametrów w MLflow oraz artefaktów takich jak krzywa ROC i macierz konfuzji.

    :param model: Wytrenowany model.
    :param metrics: Słownik z obliczonymi metrykami.
    :param run_name: Nazwa uruchomienia MLflow.
    :param params: Słownik z parametrami modelu.
    """

    # Load the input (X) and output (y) data
    X, _ = load_preprocessed_data()
    y_pred = model.predict(X)
    X_np = X.to_numpy()

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_metrics(metrics)
        if params:
            mlflow.log_params(params)

        if hasattr(model, 'save'):
            signature = infer_signature(X_np, y_pred)
            mlflow.keras.log_model(model, "model", signature=signature)

        elif hasattr(model, 'predict'):
            signature = infer_signature(X, y_pred)
            mlflow.sklearn.log_model(model, "model", signature=signature)

        else:
            raise TypeError("Nieznany typ modelu. Model musi być zgodny z API scikit-learn lub Keras.")


        # Logowanie artefaktów z folderu 'Results'
        artifacts_path = os.path.join("Results", model_name)
        if os.path.exists(artifacts_path):
            for file_name in os.listdir(artifacts_path):
                file_path = os.path.join(artifacts_path, file_name)
                mlflow.log_artifact(file_path)

        print(f"Model, metrics, and artifacts logged to MLflow run {run.info.run_id}")


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
    Evaluate the model and save the results (metrics, confusion matrix, ROC curve) to a temporary directory.
    The results will be later used to log artifacts to MLflow.

    :param model_name: Name of the model.
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param y_pred_proba: Predicted probabilities.
    :param dir: Temporary directory to save the evaluation results.
    """

    dir = 'Results'
    model_dir = os.path.join(dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    metrics['model'] = model_name  # Add model name to the metrics

    # Save metrics to a single Excel sheet
    os.makedirs(dir, exist_ok=True)
    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_file = os.path.join(dir, 'metrics.xlsx')

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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, linewidths=.5)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    cm_file = os.path.join(model_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(cm_file)
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
    roc_file = os.path.join(model_dir, f'{model_name}_roc_curve.png')
    plt.savefig(roc_file)
    plt.close()

    print(f"Confusion matrix and ROC curve of model {model_name} saved in {dir}")

    return metrics_file, cm_file, roc_file
