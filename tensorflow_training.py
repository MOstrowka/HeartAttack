import keras_tuner as kt
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import pandas as pd
import mlflow
from collections import Counter
from utils import evaluate_and_save_results, save_best_params_to_file, plot_confusion_matrix, plot_roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os


# Function to build the model using HyperParameters
def build_model(hp):
    model = Sequential()
    num_layers = hp.Int('num_layers', 1, 4)
    for i in range(num_layers):
        model.add(Dense(units=hp.Int(f'units_{i}', min_value=16, max_value=256, step=16), activation='relu'))
        model.add(Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))

    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Function to determine the most common parameters
def most_common_params(params_list):
    common_params = {}
    for key in params_list[0].keys():
        values = [params[key] for params in params_list]
        most_common_value = Counter(values).most_common(1)[0][0]
        common_params[key] = most_common_value

    num_layers = common_params['num_layers']
    filtered_params = {k: common_params[k] for k in common_params if
                       ('units_' not in k and 'dropout_' not in k) or int(k.split('_')[1]) < num_layers}

    return filtered_params


# Function to train the TensorFlow model with MLflow integration
def train_tensorflow_model(X, y):
    tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=1,
        overwrite=True,
        directory='tuner_results',
        project_name='HeartAttack'
    )

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_trials = []

    experiment_name = "KerasTuner_Optimization"
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    all_y_true = []
    all_y_pred_proba = []
    all_y_pred = []

    for fold_no, (train_index, val_index) in enumerate(kfold.split(X, y), 1):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Definicja batch_size jako hiperparametru
        hp = tuner.oracle.hyperparameters
        batch_size = hp.Int('batch_size', min_value=16, max_value=128, step=16)

        # Wywołanie tunera z przeszukiwanym batch_size
        tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=batch_size, verbose=1,
                     callbacks=[early_stopping])

        best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
        best_trials.append(best_trial)

        best_model = tuner.get_best_models(num_models=1)[0]

        # Przewidywanie bez dekorowania funkcji
        y_pred_proba = best_model.predict(X_val).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        all_y_true.extend(y_val)
        all_y_pred_proba.extend(y_pred_proba)
        all_y_pred.extend(y_pred)

    accuracy = accuracy_score(all_y_true, all_y_pred)
    precision = precision_score(all_y_true, all_y_pred)
    recall = recall_score(all_y_true, all_y_pred)
    f1 = f1_score(all_y_true, all_y_pred)
    roc_auc = roc_auc_score(all_y_true, all_y_pred_proba)

    with mlflow.start_run(run_name="Neural_Network_model"):
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        })

        # Logowanie modelu bez input_example, aby uniknąć problemów
        mlflow.keras.log_model(best_model, "model")

        os.makedirs('Results', exist_ok=True)
        plot_confusion_matrix(all_y_true, all_y_pred, title="Neural Network confusion matrix",
                              output_path=os.path.join('Results', "Neural_Network_confusion_matrix.png"))
        plot_roc_curve(all_y_true, all_y_pred_proba, title="Neural Network ROC Curve",
                       output_path=os.path.join('Results', "Neural_Network_roc_curve.png"))

        if os.path.exists(os.path.join('Results', "Neural_Network_confusion_matrix.png")):
            mlflow.log_artifact(os.path.join('Results', "Neural_Network_confusion_matrix.png"))
        if os.path.exists(os.path.join('Results', "Neural_Network_roc_curve.png")):
            mlflow.log_artifact(os.path.join('Results', "Neural_Network_roc_curve.png"))

    common_params = most_common_params([trial.hyperparameters.values for trial in best_trials])
    return best_model, common_params


def main():
    X = pd.read_csv('Data/X_preprocessed.csv').values
    y = pd.read_csv('Data/y_preprocessed.csv').values.ravel()

    best_model, best_params = train_tensorflow_model(X, y)

    y_pred_proba = best_model.predict(X).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    evaluate_and_save_results('NeuralNetwork', y, y_pred, y_pred_proba)

    save_best_params_to_file('NeuralNetwork', best_params)


if __name__ == "__main__":
    main()
