from utils import evaluate_and_save_results, load_model_configs
from Sklearn_training import train_sklearn_model
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC


def main():
    # Wczytaj dane
    X = pd.read_csv('Data/X_preprocessed.csv')
    y = pd.read_csv('Data/y_preprocessed.csv').values.ravel()

    # Wczytaj konfigurację modeli z pliku JSON
    models_params = load_model_configs('Models/model_configs.json')

    model_classes = {
        "LogisticRegression": LogisticRegression,
        "XGBClassifier": XGBClassifier,
        "SVC": SVC
    }

    # Pętla po wszystkich modelach
    for model_name, mp in models_params.items():
        print(f"Training and evaluating model: {model_name}\n")
        model_class = model_classes.get(mp['model'])
        if model_class is None:
            raise ValueError(f"Model class {mp['model']} is not defined in model_classes.")

        model = model_class(random_state=42, probability=True) if 'SVC' in model_name else model_class(random_state=42)
        params = mp['params']

        # Trenuj model i uzyskaj najlepsze parametry
        best_model, y_pred, y_pred_proba, model_name = train_sklearn_model(model, X, y, params=params, cv=5)

        # Ewaluacja i zapis wyników
        evaluate_and_save_results(model_name, y, y_pred, y_pred_proba)


if __name__ == "__main__":
    main()
