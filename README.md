
![Heart Attack Image](https://www.heart.org/-/media/Images/News/2021/June-2021/0623SilentHeartAttack_SC.jpg)

# Heart Attack Prediction

This project focuses on predicting the risk of heart attack using a dataset that contains various health-related parameters.

## Project Structure

### Dataset Description

This dataset is related to diagnosing heart disease and contains 296 records, each with the following variables:

- **Age**: Age of the patient (integer, in years).
- **Sex**: Gender of the patient (categorical).
- **ChestPainType**: Type of chest pain (categorical).
- **RestingBloodPressure**: Resting blood pressure upon admission (in mm Hg).
- **Cholesterol**: Serum cholesterol level (mg/dl).
- **FastingBloodSugar**: Fasting blood sugar level (whether greater than 120 mg/dl, categorical).
- **RestingECG**: Resting electrocardiogram results (categorical).
- **MaxHeartRate**: Maximum heart rate achieved (integer).
- **ExerciseInducedAngina**: Exercise-induced angina (categorical).
- **STDepression**: ST depression induced by exercise relative to rest (integer).
- **STSlope**: Slope of the ST segment (categorical).
- **NumMajorVessels**: Number of major vessels (0-3) colored by fluoroscopy (integer).
- **Thalassemia**: Presence of thalassemia (categorical).

The project is organized into several folders and files, each serving a specific purpose:

### 1. `Data/`
- **DataRaw.csv**: The raw dataset used for training and evaluation.
- **X_preprocessed.csv**: The preprocessed feature set used for model training.
- **y_preprocessed.csv**: The preprocessed target labels corresponding to the features.

### 2. `EDA/`
This folder contains files related to Exploratory Data Analysis (EDA), including:
- **stats.xlsx**: Descriptive statistics of the dataset.
- **CorrMatrix.png**: A heatmap showing the correlation matrix for numerical features.
- **Countplots.png**: Count plots for categorical variables.
- **distribution.xlsx**: Analysis results including skewness, kurtosis, and Shapiro-Wilk test statistics for numerical features.
- **Histograms_and_Boxplots.png**: Histograms and box plots for numerical features.
- **Pairplots.png**: Pair plots showing relationships between numerical features, with differentiation based on heart attack risk.

### 3. `Models/`
This folder contains configurations and results of trained models:
- **model_configs.json**: A configuration file defining the models and their hyperparameter grids used during training.

#### `Models/BestParams/`
This subfolder contains JSON files that store the best hyperparameters for each model after tuning.

#### `Models/SavedModels/`
This subfolder includes the serialized versions of the trained models, saved either in `pkl` format (for scikit-learn models) or `keras` format (for the Neural Network).

### 4. `Results/`
The `Results` folder holds the evaluation metrics and visualizations for all trained models, including:
- Summary of model performance metrics in Excel format.
- Confusion matrices showing the performance of each model in classifying the data.
- ROC curves demonstrating the true positive rate against the false positive rate for each model.

## Scripts Description

The project includes several Python scripts that handle different stages of the machine learning pipeline:

- **`DataPreprocess.py`**: Handles data cleaning, exploratory data analysis (EDA), and data preprocessing. It outputs the cleaned and preprocessed datasets that are used for model training.

- **`Sklearn_training.py`**: Contains functions for training scikit-learn models, including Logistic Regression, Support Vector Classifier (SVC), and XGBoost. It also performs hyperparameter tuning using `GridSearchCV` and logs the results to MLflow.

- **`Sklearn_evaluation.py`**: Uses the functions from `Sklearn_training.py` to train and evaluate scikit-learn models. It saves the best models and logs their performance metrics.

- **`Tensorflow_training.py`**: Trains a Neural Network using TensorFlow and Keras Tuner for hyperparameter optimization. The model is trained using stratified K-fold cross-validation, and the best model is saved and logged to MLflow.

- **`utils.py`**: A utility script that contains helper functions used across the project, such as loading data, calculating metrics, and logging results to MLflow.

- **`main.py`**: The main entry point for running the entire pipeline. It sequentially runs data preprocessing, scikit-learn model training and evaluation, and TensorFlow model training and evaluation.

## How to Run the Project

To run the entire project, execute the `main.py` script. This script will automatically perform the following steps:

1. **Data Preprocessing**: The raw data is cleaned, analyzed, and preprocessed.
2. **Scikit-learn Model Training and Evaluation**: Logistic Regression, SVC, and XGBoost models are trained, evaluated, and their results are logged.
3. **TensorFlow Model Training and Evaluation**: A Neural Network is trained with Keras Tuner for hyperparameter optimization, and the results are logged.

```bash
python main.py
```

Make sure that you have installed all necessary dependencies listed in `requirements.txt`, and that your environment is set up to run TensorFlow, scikit-learn, and MLflow.

## Summary

The project employs multiple machine learning models to predict the risk of heart attack, including Logistic Regression, Support Vector Classifier (SVC), XGBoost, and a Neural Network. After evaluating the performance of all models, the Neural Network was found to be the most effective, achieving the highest accuracy (91.5%) and the best performance metrics (see `metrics.xlsx` for details).

The metrics were calculated as the average of 5-fold cross-validation (for scikit-learn models) or 5-fold KFold cross-validation (for TensorFlow).

The hyperparameters for the Neural Network were optimized using Keras Tuner, which allowed for a thorough search across a range of possible configurations, resulting in the best-performing model.
