import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from DataPreprocess import main as preprocess_data
from Sklearn_evaluation import main as evaluate_models
from tensorflow_training import main as train_and_evaluate_tensorflow

if __name__ == "__main__":
    preprocess_data('Data/DataRaw.csv')
    # evaluate_models()
    train_and_evaluate_tensorflow()
