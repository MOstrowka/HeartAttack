
![Heart Attack Image](https://www.heart.org/-/media/Images/News/2021/June-2021/0623SilentHeartAttack_SC.jpg)

# Heart Attack Prediction

This project focuses on predicting the risk of heart attack using a dataset that contains various health-related parameters. The project explores different machine learning models and evaluates their performance to determine the most effective approach for this task.

## Project Structure

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

## Summary

The project employs multiple machine learning models to predict the risk of heart attack, including Logistic Regression, Support Vector Classifier (SVC), XGBoost, and a Neural Network. After evaluating the performance of all models, the Neural Network was found to be the most effective, achieving the highest accuracy (91.5%) and the best performance metrics (see metrics.xlsx for details).

The metrics were calculated as the average of 5-fold cross-validation (for scikit-learn models) or 5-fold KFold cross-validation (for TensorFlow).

The hyperparameters for the Neural Network were optimized using Keras Tuner, which allowed for a thorough search across a range of possible configurations, resulting in the best-performing model.
