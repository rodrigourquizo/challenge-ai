# 💳 Challenge AI: Bank Transaction Fraud Detection
This repository contains the implementation of a credit risk classification model developed using Amazon SageMaker and Amazon Bedrock.

[![Watch the development overview](https://img.shields.io/badge/Watch%20on-YouTube-red?logo=youtube&logoColor=white)](https://youtu.be/wYHicxTEm5Y)

## 📊 Workflow Architecture

![Architecture](https://github.com/rodrigourquizo/challenge-ai/blob/3da583fac0637f891964b6c70e3112e33877c245/workflow.png)

## 🎯 Objective
Build an intelligent, end-to-end pipeline capable of:

-Generating credit risk descriptions using generative models like GPT or LLaMA through AWS Bedrock.

-Classifying descriptions as “good risk” or “bad risk”, again using Bedrock models.

-Training a supervised XGBoost classifier in Amazon SageMaker using the generated labels.

-Deploying the trained model to SageMaker for real-time inference.

-Ensuring the system is scalable, efficient, and adaptable to new data for consistent performance.

## 📁 Project Structure

```
CHALLENGE/
├── model/
│   └── xgboost-model/                  # Trained XGBoost model artifacts
│
├── credit_risk_reto.csv.xlsx          # Original dataset
├── credit_risk_labeled.xlsx           # Dataset with AI-generated labels
├── credit_risk_with_descriptions.xlsx # Dataset with Bedrock-generated descriptions
├── credit_risk_processed_train.csv    # Preprocessed training dataset
├── credit_risk_processed_test.csv     # Preprocessed test dataset
│
├── inference.py                       # Script to call the deployed SageMaker model
└── test.ipynb                         # Notebook for testing
```

## 🧪 Dataset Overview

The dataset contains **1,000 samples** with 9 categorical and numerical features:

| Column            | Description                                   |
|-------------------|-----------------------------------------------|
| `Age`             | Customer’s age                                |
| `Sex`             | Gender                                        |
| `Job`             | Job level (0 = low skill, 3 = high skill)     |
| `Housing`         | Type of housing                               |
| `Saving accounts` | Type of savings account                       |
| `Checking account`| Type of checking account                      |
| `Credit amount`   | Requested loan amount                         |
| `Duration`        | Loan duration in months                       |
| `Purpose`         | Purpose of the loan                           |

### 🆕 Additional Columns

| Column        | Description                                         |
|----------------|-----------------------------------------------------|
| `description` | AI-generated textual description of credit risk      |
| `target`      | Risk classification (`good risk` / `bad risk`)       |

## 🚀 Workflow Summary

1. **Generate descriptions**  
   Use Amazon Bedrock LLMs (e.g., Claude, LLaMA) to generate natural-language descriptions for each credit record.

2. **Classify descriptions**  
   Use Bedrock classification prompts to assign risk labels (`good risk` or `bad risk`) based on the generated text.

3. **Train a model with SageMaker**  
   Train an XGBoost classifier using the labeled dataset (features + target) on Amazon SageMaker, including hyperparameter tuning.

4. **Deploy the model**  
   Deploy the trained model to a SageMaker real-time endpoint for serving predictions.

5. **Run inference**  
   Use the `inference.py` script to query the deployed model and get predictions on new data.

📈 **Model Performance**

| Metric         | Value  |
|----------------|--------|
| Accuracy       | 0.72   |
| Precision      | 0.77   |
| Recall         | 0.81   |
| F1-Score       | 0.79   |
| AUC-ROC        | 0.74   |

🔢 **Confusion Matrix**

|                | Predicted: Bad Risk | Predicted: Good Risk |
|----------------|---------------------|----------------------|
| **Actual: Bad Risk**   | 40                  | 31                   |
| **Actual: Good Risk**  | 25                  | 104                  |

## 🛠 Requirements

- **Python 3.8+**
- **An AWS account** with:
  - Access to **Amazon Bedrock**
  - Permissions for **Amazon SageMaker** (especially `InvokeEndpoint`)
- **Python libraries**:
  - `boto3`
  - `sagemaker`
  - `pandas`



