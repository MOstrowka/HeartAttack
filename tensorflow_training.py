import keras_tuner as kt
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import pandas as pd
from collections import Counter
from utils import evaluate_and_save_results, save_best_params_to_file
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# Funkcja budowania modelu z użyciem HyperParameters
def build_model(hp):
    model = Sequential()

    # Liczba ukrytych warstw
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(Dense(units=hp.Int(f'units_{i}', min_value=8, max_value=256, step=8), activation='relu'))
        model.add(Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)))

    # Warstwa wyjściowa
    model.add(Dense(1, activation='sigmoid'))

    # Learning rate dla optymalizatora Adam
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    # Kompilacja modelu
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# Funkcja trenowania modelu tensorflow
def train_tensorflow_model(X, y):
    tuner = kt.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=20,
        executions_per_trial=1,
        overwrite=True,
        directory='tuner_results',  # Dodaj katalog dla wyników tunera
    )

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_params_per_fold = []
    results = []

    for fold_no, (train_index, val_index) in enumerate(kfold.split(X, y), 1):
        print(f"Fold {fold_no}")

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Dodajemy callback EarlyStopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=50, verbose=0, callbacks=[early_stopping])

        best_model = tuner.get_best_models(num_models=1)[0]
        best_hp = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values
        best_params_per_fold.append(best_hp)

        # Pobierz liczbę epok, po których osiągnięto najlepszy wynik
        history = best_model.history.history
        best_epoch = np.argmin(history['val_loss']) + 1  # +1, aby było liczone od 1
        best_params_per_fold[-1]['best_epoch'] = best_epoch

        y_pred_proba = best_model.predict(X_val).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_pred_proba)

        print(f"Fold {fold_no} results - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, ROC AUC: {roc_auc}")
        results.append([accuracy, precision, recall, f1, roc_auc])

    mean_results = np.mean(results, axis=0)
    return best_model, mean_results, best_params_per_fold


def main():
    # Wczytaj dane
    X = pd.read_csv('Data/X_preprocessed.csv').values
    y = pd.read_csv('Data/y_preprocessed.csv').values.ravel()

    best_model, mean_results, best_params = train_tensorflow_model(X, y)

    # Ewaluacja i zapis wyników
    y_pred_proba = best_model.predict(X).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    evaluate_and_save_results('NeuralNetwork', y, y_pred, y_pred_proba)

    # Zapisywanie najlepszych parametrów do pliku JSON
    save_best_params_to_file('NeuralNetwork', best_params)


if __name__ == "__main__":
    main()
