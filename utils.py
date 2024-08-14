import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_metrics(y_true, y_pred, y_pred_proba):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'ROC AUC': roc_auc_score(y_true, y_pred_proba)
    }
    return metrics

def evaluate_and_save_results(model_name, y, y_pred, y_pred_proba, metrics_df_path='Results/metrics.xlsx'):
    os.makedirs('Results', exist_ok=True)

    metrics = calculate_metrics(y, y_pred, y_pred_proba)

    if os.path.exists(metrics_df_path):
        metrics_df = pd.read_excel(metrics_df_path)
    else:
        metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])

    new_metrics = pd.DataFrame({
        'Model': [model_name],
        **metrics
    })

    new_metrics = new_metrics.dropna(axis=1, how='all')
    metrics_df = metrics_df.dropna(axis=1, how='all')

    metrics_df = pd.concat([metrics_df, new_metrics], ignore_index=True)

    metrics_df.to_excel(metrics_df_path, index=False)

    print(f"Metrics of model {model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"Metrics saved to {metrics_df_path}")
    print("-" * 50)

def save_best_params_to_file(model_name, best_params, output_dir='Models/BestParams'):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{model_name}_best_params.json')

    data = {
        model_name: {
            "model": model_name,
            "params": best_params
        }
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Best parameters of model {model_name} saved to {output_path}")

def load_model_configs(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
