import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.linalg import pinv

# Function to compute the linear kernel matrix
def compute_linear_kernel(X, X2):
    # Calculate the linear kernel matrix
    return np.dot(X, X2.T)

# Function to train KELM with regularization
def train_kelm(X_train, y_train, reg_coeff):
    K_train = compute_linear_kernel(X_train, X_train)
    K_train = np.hstack((np.ones((K_train.shape[0], 1)), K_train))
    # Apply regularization
    identity_matrix = np.eye(K_train.shape[1])
    pseudo_inverse = pinv(K_train.T @ K_train + reg_coeff * identity_matrix) @ K_train.T
    weights = pseudo_inverse @ y_train
    return weights

# Function to predict with KELM
def predict_kelm(X_train, X_test, weights):
    K_test = compute_linear_kernel(X_test, X_train)
    K_test = np.hstack((np.ones((K_test.shape[0], 1)), K_test))
    y_pred = K_test @ weights
    return np.argmax(y_pred, axis=1)

# Load folds from the directory
output_dir = 'E:\\BISMILLAHIRAHMANIRAHIM SKRIPSI\\KFold_Folds'

# Regularization coefficients to test
regularization_coeffs = [0.000000000000000000000000000001, 0.000000000000000000001, 0.0000000000000010, 0.0000000000100]

# One-hot encode target if needed
encoder = OneHotEncoder(sparse=False)

# Create a Pandas Excel writer using XlsxWriter as the engine.
excel_file_path = 'E:\\BISMILLAHIRAHMANIRAHIM SKRIPSI\\Linear_KELM_Evaluation_Results.xlsx'
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
            weights = train_kelm(X_train, y_train_encoded, reg_coeff)

            # Predict the test set using the trained KELM model
            y_pred_encoded = predict_kelm(X_train, X_test, weights)
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
