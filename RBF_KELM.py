import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold

# Load the CSV file from local directory
data_path = 'E:\\BISMILLAHIRAHMANIRAHIM SKRIPSI\\KesehatanMaternalIsolation.csv'
data = pd.read_csv(data_path)

# Display the data
print(data.head())
print(data.describe())
print(data.info())

# Check for missing values in variables
print(data.isnull().sum())

# Ensure 'RiskLevel' column exists in the dataset
if 'RiskLevel' not in data.columns:
    raise KeyError("The 'RiskLevel' column is not found in the dataset. Ensure the dataset has the 'RiskLevel' column.")

# Convert 'RiskLevel' from string to numeric
risk_mapping = {'low risk': 0, 'mid risk': 1, 'high risk': 2}
data['RiskLevel'] = data['RiskLevel'].map(risk_mapping)

# Convert relevant columns to numeric and handle errors
data['BS'] = pd.to_numeric(data['BS'], errors='coerce')
data['BodyTemp'] = pd.to_numeric(data['BodyTemp'], errors='coerce')

# Handle missing values: fill NaN in 'BodyTemp' with the column's mean
data['BodyTemp'].fillna(data['BodyTemp'].mean(), inplace=True)

# Check for missing values again
print(data.isnull().sum())

# Select columns to normalize
columns_to_normalize = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Apply MinMaxScaler to the selected columns
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

# Display the normalized data
print("Data after normalization:")
print(data[columns_to_normalize].head())

# KFold cross-validation with 10 folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Directory to save the folds
output_dir = 'E:\\BISMILLAHIRAHMANIRAHIM SKRIPSI\\KFold_Folds'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Split data into 10 folds and save each fold as .csv and .xlsx
fold_number = 1
for train_index, test_index in kf.split(data):
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]
    
    # Define file paths for saving
    train_file_path_csv = os.path.join(output_dir, f'fold_{fold_number}_train.csv')
    test_file_path_csv = os.path.join(output_dir, f'fold_{fold_number}_test.csv')
    train_file_path_xlsx = os.path.join(output_dir, f'fold_{fold_number}_train.xlsx')
    test_file_path_xlsx = os.path.join(output_dir, f'fold_{fold_number}_test.xlsx')
    
    # Save the training and testing data as .csv files
    train_data.to_csv(train_file_path_csv, index=False)
    test_data.to_csv(test_file_path_csv, index=False)
    
    # Save the training and testing data as .xlsx files
    train_data.to_excel(train_file_path_xlsx, index=False)
    test_data.to_excel(test_file_path_xlsx, index=False)
    
    print(f"Fold {fold_number}: Training and test datasets saved as .csv and .xlsx.")
    
    fold_number += 1


#-----------------------------------------------------------------------------------------------------------------------------------------------
#__________________________________________________________________________________________________________________________________________________________

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.linalg import pinv

# Function to compute the RBF kernel matrix
def compute_rbf_kernel(X, X2, gamma=0.5):
    # Calculate the squared Euclidean distance between X and X2
    sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X, X2.T)
    return np.exp(-gamma * sq_dists)

# Function to train KELM with regularization
def train_kelm(X_train, y_train, reg_coeff, gamma=0.5):
    K_train = compute_rbf_kernel(X_train, X_train, gamma)
    K_train = np.hstack((np.ones((K_train.shape[0], 1)), K_train))
    # Apply regularization = koefisien regulasinya
    identity_matrix = np.eye(K_train.shape[1])
    pseudo_inverse = pinv(K_train.T @ K_train + reg_coeff * identity_matrix) @ K_train.T
    weights = pseudo_inverse @ y_train
    return weights

# Function to predict with KELM
def predict_kelm(X_train, X_test, weights, gamma=0.5):
    K_test = compute_rbf_kernel(X_test, X_train, gamma)
    K_test = np.hstack((np.ones((K_test.shape[0], 1)), K_test))
    y_pred = K_test @ weights
    return np.argmax(y_pred, axis=1)

# Load folds from the directory
output_dir = 'E:\\BISMILLAHIRAHMANIRAHIM SKRIPSI\\KFold_Folds'

# Regularization coefficients to test
regularization_coeffs = [0.1, 1, 10, 100]

# One-hot encode target if needed
encoder = OneHotEncoder(sparse=False)

# Create a Pandas Excel writer using XlsxWriter as the engine.
excel_file_path = 'E:\\BISMILLAHIRAHMANIRAHIM SKRIPSI\\RBF_KELM_Evaluation_Results.xlsx'
with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
    # Loop through regularization coefficients
    for reg_coeff in regularization_coeffs:
        print(f"\nEvaluating for Regularization Coefficient: {reg_coeff}")

        # Create a DataFrame to store the results for this regularization coefficient
        fold_results = []

        # Iterate through the 10 folds
        for fold_number in range(1, 11):
            # Load train and test data for the current fold
            train_file_path = os.path.join(output_dir, f'fold_{fold_number}_train.csv')
            test_file_path = os.path.join(output_dir, f'fold_{fold_number}_test.csv')

            # Read the CSV files for this fold
            train_data = pd.read_csv(train_file_path)
            test_data = pd.read_csv(test_file_path)

            # Separate input features and target
            X_train = train_data[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']].values
            y_train = train_data[['RiskLevel']].values

            X_test = test_data[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']].values
            y_test = test_data[['RiskLevel']].values

            # One-hot encode the target labels
            y_train_encoded = encoder.fit_transform(y_train)
            y_test_encoded = encoder.transform(y_test)

            # Train KELM with the current regularization coefficient
            gamma = 0.5  # Assuming default for now, could be tuned later
            weights = train_kelm(X_train, y_train_encoded, reg_coeff, gamma)

            # Predict the test set using the trained KELM model
            y_pred_encoded = predict_kelm(X_train, X_test, weights, gamma)
            y_pred = encoder.inverse_transform(np.eye(y_test_encoded.shape[1])[y_pred_encoded])

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Calculate Sensitivity and Specificity from the confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            sensitivity = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
            specificity = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)

            # Store results for this fold
            fold_results.append({
                'Fold': fold_number,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'Sensitivity': np.nanmean(sensitivity),  # Use np.nanmean to avoid NaN issues
                'Specificity': np.nanmean(specificity)   # Use np.nanmean to avoid NaN issues
            })

        # Convert results to DataFrame
        results_df = pd.DataFrame(fold_results)

        # Write the DataFrame to a new sheet in the Excel file
        results_df.to_excel(writer, sheet_name=f'Regularization_{reg_coeff}', index=False)

        print(f"Results for Regularization Coefficient {reg_coeff} written to Excel.")

print("All results saved to Excel file successfully.")
