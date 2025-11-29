import pandas as pd
import numpy as np
import boto3
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from . import config
from . import processing

def run_inference(input_file):
    """Runs inference pipeline."""
    print(f"Loading data from {input_file}...")
    if input_file.endswith('.xlsx'):
        df = pd.read_excel(input_file)
    else:
        df = pd.read_csv(input_file)

    print("Preprocessing data...")
    prep_df = processing.preprocess_data(df)

    print(f"Invoking endpoint {config.SAGEMAKER_ENDPOINT_NAME}...")
    predictor = Predictor(endpoint_name=config.SAGEMAKER_ENDPOINT_NAME, serializer=CSVSerializer())

    X = prep_df.iloc[:, 1:].values  # All except first column (target)
    y = prep_df.iloc[:, 0].values   # First column is target

    response = predictor.predict(X)
    predictions = np.fromstring(response.decode('utf-8'), sep='\n')
    binary_predictions = (predictions >= 0.5).astype(int)

    # Metrics
    print("Calculating metrics...")
    # Handle case where y might be all NaNs if we don't have ground truth
    if not np.isnan(y).all():
        accuracy = accuracy_score(y, binary_predictions)
        precision = precision_score(y, binary_predictions, zero_division=0)
        recall = recall_score(y, binary_predictions, zero_division=0)
        f1 = f1_score(y, binary_predictions, zero_division=0)
        try:
            auc = roc_auc_score(y, predictions)
        except ValueError:
            auc = 0.0 # Handle case with only one class
        conf_matrix = confusion_matrix(y, binary_predictions)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")
        print(f"AUC-ROC: {auc:.2f}")
        print("Confusion Matrix:\n", conf_matrix)
    else:
        print("No ground truth labels found. Returning predictions.")
    
    return predictions, binary_predictions

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = config.RAW_DATA_PATH # Default to raw data
    
    run_inference(input_path)
