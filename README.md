# 💳 Challenge AI: Bank Transaction Fraud Detection
This repository contains the implementation of a credit risk classification model developed using Amazon SageMaker and Amazon Bedrock.

🎯 Objective
Build an intelligent, end-to-end pipeline capable of:
-Generating credit risk descriptions using generative models like GPT or LLaMA through AWS Bedrock.
-Classifying descriptions as “good risk” or “bad risk”, again using Bedrock models.
-Training a supervised XGBoost classifier in Amazon SageMaker using the generated labels.
-Deploying the trained model to SageMaker for real-time inference.
-Ensuring the system is scalable, efficient, and adaptable to new data for consistent performance.

📁 Project Structure
CHALLENGE/
│
├── model/
│   └── xgboost-model/                  # Folder with trained XGBoost model artifacts
│
├── credit_risk_reto.csv.xlsx          # Original dataset
├── credit_risk_labeled.xlsx           # Descriptions + generated labels
├── credit_risk_with_descriptions.xlsx # Only Bedrock-generated descriptions
├── credit_risk_processed_train.csv    # Preprocessed training set
├── credit_risk_processed_test.csv     # Preprocessed test set
│
├── inference.py                       # Script to invoke deployed SageMaker endpoint
├── test.ipynb                         # Notebook for local testing and experimentation
