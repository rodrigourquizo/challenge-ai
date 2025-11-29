import os

# AWS Configuration
AWS_REGION = 'us-east-1'
BEDROCK_MODEL_ID = 'arn:aws:bedrock:us-east-1:004082821794:inference-profile/us.meta.llama3-2-90b-instruct-v1:0'
SAGEMAKER_ENDPOINT_NAME = 'sagemaker-xgboost-2025-07-24-03-20-18-825'

# Data Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'credit_risk_reto.xlsx')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Model Configuration
TARGET_COLUMN = 'target'
NUMERIC_COLUMNS = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration']
CATEGORICAL_COLUMNS = ['Purpose']
