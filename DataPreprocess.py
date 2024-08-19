import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, skew, kurtosis
from sklearn.preprocessing import StandardScaler
import os


def load_and_clean_data(data_path):
    # Load data
    df = pd.read_csv(data_path)

    # Rename columns
    df.columns = ['Age', 'Sex', 'ChestPainType', 'RestingBloodPressure', 'Cholesterol', 'FastingBloodSugar',
                  'RestingECG', 'MaxHeartRate', 'ExerciseInducedAngina', 'STDepression', 'STSlope',
                  'NumMajorVessels', 'Thalassemia', 'HeartAttackRisk']


    print("Removing wrong data..\n")
    start = len(df)

    # Drop incorrect values
    df = df[df['NumMajorVessels'] < 4]  # Drop the wrong NumMajorVessels values
    df = df[df['Thalassemia'] > 0]  # Drop the wrong Thalassemia values
    df = df.reset_index(drop=True)  # Reset the index

    print(f'The length of the data now is {len(df)} instead of {start}\n')

    # Convert to float64
    df = df.astype('float64')

    # Save the statistics to an Excel file
    stats_output_dir = "EDA"
    os.makedirs(stats_output_dir, exist_ok=True)

    stats_file_path = os.path.join(stats_output_dir, "stats.xlsx")
    df.describe().transpose().to_excel(stats_file_path)
    print(f"Descriptive statistics saved to {stats_file_path}\n")

    return df


def perform_eda(df, output_dir="EDA"):
    os.makedirs(output_dir, exist_ok=True)

    numerical_columns = ['Age', 'RestingBloodPressure', 'Cholesterol', 'MaxHeartRate', 'STDepression']

    # EDA: Histograms and Boxplots
    plt.figure(figsize=(15, 20))
    for i, col in enumerate(numerical_columns):
        # Histogram
        plt.subplot(len(numerical_columns), 2, 2 * i + 1)
        sns.histplot(df[col], kde=True, bins=30, alpha=0.7)
        plt.grid(color='#000000', linestyle=':', zorder=0, dashes=(1, 5))
        plt.title(f'{col}')

        # Boxplot
        plt.subplot(len(numerical_columns), 2, 2 * i + 2)
        sns.boxplot(x=df[col], color="skyblue", fliersize=5, linewidth=1.5)
        plt.grid(color='#000000', linestyle=':', zorder=0, dashes=(1, 5))
        plt.title(f'{col}')

    plt.tight_layout(pad=2.0)
    hist_boxplot_file = os.path.join(output_dir, "Histograms_and_Boxplots.png")
    plt.savefig(hist_boxplot_file, dpi=1000)
    plt.close()
    print(f"Histograms and boxplots saved to {hist_boxplot_file}\n")

    # Distribution values analysis
    results = []
    for col in numerical_columns:
        stat, p_value = shapiro(df[col])
        skewness = skew(df[col])
        kurt = kurtosis(df[col])

        results.append({
            'Column': col,
            'Shapiro-Wilk Statistic': stat,
            'p-value': p_value,
            'Skewness': skewness,
            'Kurtosis': kurt
        })

    results_df = pd.DataFrame(results)

    # Save distribution results to an Excel file
    distribution_file_path = os.path.join(output_dir, "distribution.xlsx")
    results_df.to_excel(distribution_file_path, index=False)
    print(f"Distribution analysis results saved to {distribution_file_path}\n")

    categorical_columns = ['Sex', 'ChestPainType', 'FastingBloodSugar', 'RestingECG',
                           'ExerciseInducedAngina', 'STSlope', 'NumMajorVessels', 'Thalassemia', 'HeartAttackRisk']

    # EDA: Countplots
    plt.figure(figsize=(15, 15))
    for i, col in enumerate(categorical_columns, 1):
        plt.subplot(3, 3, i)
        sns.countplot(x=df[col], hue=df[col], palette='Set2', legend=False)
        plt.grid(color='#000000', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
        plt.title(f'Distribution of {col}')

    plt.tight_layout(pad=2.0)
    countplot_file = os.path.join(output_dir, "Countplots.png")
    plt.savefig(countplot_file, dpi=1000)
    plt.close()
    print(f"Countplots saved to {countplot_file}\n")

    # Correlation matrix
    correlation_matrix = df[numerical_columns].corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    plt.figure(figsize=(8, 6))
    corr_matrix_file = os.path.join(output_dir, "CorrMatrix.png")
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', square=True, fmt='.2f')
    plt.title('Correlation Matrix for Numerical Features', fontsize=12)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(corr_matrix_file, dpi=300)
    plt.close()
    print(f"Correlation matrix saved to {corr_matrix_file}")

    # Pairplots
    pairplot_file = os.path.join(output_dir, "Pairplots.png")
    sns.pairplot(df, hue='HeartAttackRisk', vars=numerical_columns, palette='Set1')
    plt.savefig(pairplot_file, dpi=1000)
    plt.close()
    print(f"Pairplots saved to {pairplot_file}\n")


def preprocess_data(df, output_dir="Data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X = df.drop(['HeartAttackRisk'], axis=1)
    y = df['HeartAttackRisk']

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = y.to_numpy()

    # Save preprocessed data with column names for X
    X_file_path = os.path.join(output_dir, 'X_preprocessed.csv')
    y_file_path = os.path.join(output_dir, 'y_preprocessed.csv')

    # Here, we use the original column names
    pd.DataFrame(X_scaled, columns=X.columns).to_csv(X_file_path, index=False)
    pd.DataFrame(y, columns=['HeartAttackRisk']).to_csv(y_file_path, index=False)

    print(f"Preprocessing complete. Preprocessed data saved to {X_file_path} and {y_file_path}\n")


def main(data_path='Data/DataRaw.csv'):
    df = load_and_clean_data(data_path)
    perform_eda(df)
    preprocess_data(df)


if __name__ == "__main__":
    main()
