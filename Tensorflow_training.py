import os
import keras_tuner as kt
import mlflow
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import Counter
import tempfile
from utils import (
    evaluate_and_save_results,
    save_best_params_to_file,
    calculate_metrics,
    log_to_mlflow,
    load_preprocessed_data
)

def build_model(hp):
    """
    Build a Keras model with hyperparameters.

    :param hp: Hyperparameters from Keras Tuner.
    :return: Compiled Keras model.
    """
    model = Sequential()
    num_layers = hp.Int('num_layers', 1, 4)
    for i in range(num_layers):
        model.add(Dense(units=hp.Int(f'units_{i}', min_value=16, max_value=256, step=16), activation='relu'))
        model.add(Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))

    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    return model

def most_common_params(params_list):
    """
    Determine the most common parameters from a list of parameter sets.

    :param params_list: List of parameter dictionaries.
    :return: Dictionary of most common parameters.
    """
    common_params = {}
    for key in params_list[0].keys():
        values = [params[key] for params in params_list]
        most_common_value = Counter(values).most_common(1)[0][0]
        common_params[key] = most_common_value

    num_layers = common_params['num_layers']
    filtered_params = {
        key: value for key, value in common_params.items()
        if ('units_' not in key and 'dropout_' not in key) or int(key.split('_')[1]) < num_layers
    }

    return filtered_params

def train_tensorflow_model(X, y):
    """
    Train a TensorFlow model using Keras Tuner and Stratified K-Fold cross-validation.

    :param X: Features.
    :param y: Labels.
    :return: Best trained model and its parameters.
    """
    best_model = None

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
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        hp = tuner.oracle.hyperparameters
        batch_size = hp.Int('batch_size', min_value=16, max_value=128, step=16)

        tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=batch_size, verbose=1,
                     callbacks=[early_stopping])

        best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
        best_trials.append(best_trial)

        best_model = tuner.get_best_models(num_models=1)[0]

        y_pred_proba = best_model.predict(X_val).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        all_y_true.extend(y_val)
        all_y_pred_proba.extend(y_pred_proba)
        all_y_pred.extend(y_pred)

    metrics = calculate_metrics(all_y_true, all_y_pred, all_y_pred_proba)
    common_params = most_common_params([trial.hyperparameters.values for trial in best_trials])
    log_to_mlflow(best_model, metrics, run_name="Neural_Network_model", params=common_params)

    # Save the best model in .keras format, without including optimizer
    model_save_dir = 'Models/SavedModels'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        print(f"Directory {model_save_dir} created.")
    else:
        print(f"Directory {model_save_dir} already exists.")

    model_save_path = os.path.join(model_save_dir, 'NeuralNetwork_best_model.keras')
    best_model.save(model_save_path, include_optimizer=False)
    print(f"Model saved to {model_save_path}")

    # Ensure the directory exists before saving input example
    temp_dir = tempfile.mkdtemp()
    mlflow.keras.log_model(best_model, artifact_path=os.path.join(temp_dir, "NeuralNetwork_best_model"))

    return best_model, common_params

def main():
    """
    Main function to execute the TensorFlow model training.
    """
    X, y = load_preprocessed_data()

    best_model, best_params = train_tensorflow_model(X, y)

    y_pred_proba = best_model.predict(X).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    evaluate_and_save_results('NeuralNetwork', y, y_pred, y_pred_proba)

    save_best_params_to_file('NeuralNetwork', best_params)

if __name__ == "__main__":
    main()
