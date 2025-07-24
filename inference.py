import pandas as pd
import boto3
import json
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
import numpy as np

def extraer_target(descripcion):
    if 'good risk' in descripcion.lower():
        return 'good risk'
    if 'buen riesgo' in descripcion.lower():
        return 'good risk'
    if 'bad risk' in descripcion.lower():
        return 'bad risk'
    if 'mal riesgo' in descripcion.lower():
        return 'bad risk'


def preprocess(df):
    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

    #Generacion de descripciones
    descriptions = []
    for i in range(len(df)):
        fila = df.iloc[i]  
        
        ahorros = fila['Saving accounts'] if fila['Saving accounts'] else 'desconocido'
        cuenta_corriente = fila['Checking account'] if fila['Checking account'] else 'desconocido'
        mapa_trabajo = {0: 'no cualificado no residente', 1: 'no cualificado residente', 2: 'cualificado', 3: 'altamente cualificado'}
        desc_trabajo = mapa_trabajo.get(fila['Job'], 'desconocido')

        #Crear prompt
        prompt = f"""
        Genera UNA descripción concisa del perfil de riesgo crediticio para una persona con estas características en UNA linea, DETENIÉNDOTE después de aproximadamente 50 palabras:
        - Edad: {fila['Age']}
        - Sexo: {fila['Sex']}
        - Trabajo: {desc_trabajo}
        - Vivienda: {fila['Housing']}
        - Cuentas de ahorro: {ahorros}
        - Cuenta corriente: {cuenta_corriente}
        - Monto de crédito: {fila['Credit amount']} EUR
        - Duración: {fila['Duration']} meses
        - Propósito: {fila['Purpose']}

        Evalúa el riesgo crediticio en solo un párrafo corto, sin repeticiones ni preguntas, ni contenido adicional
        Al final indica si es “bad risk” o “good risk”,asegurando que la evaluación sea coherente con la clasificación, solo una de esas dos opciones, sin explicaciones adicionales.
        Si el riesgo crediticio es "moderado a alto" se tiene que clasificar como "bad risk" a menos que haya factores compensatorios fuertes (e.g., ahorros significativos).

        """

        body = json.dumps({
            "prompt": prompt,
            "temperature": 0.6,
            #"max_tokens": 100
        })
        try:
            response = bedrock.invoke_model(
                modelId='arn:aws:bedrock:us-east-1:004082821794:inference-profile/us.meta.llama3-2-90b-instruct-v1:0',
                contentType='application/json',
                accept='application/json',
                body=body
            )
            response_body = json.loads(response['body'].read())
            
            if 'generation' in response_body: 
                description = response_body['generation'].strip()
            else:
                description = "No se pudo extraer la descripción."
            
            descriptions.append(description)
        except Exception as e:
            print(f"Error en fila {i}: {e}")
            descriptions.append("No se pudo generar la descripción.")

    df.loc[:, 'description'] = descriptions

    #Etiquetado
    df['target'] = df['description'].apply(extraer_target)

    #Preprocesamiento para modelo en SageMaker
    df['target'] = df['target'].map({'good risk': 0, 'bad risk': 1})

    if df['Sex'].dtype == 'object':
        df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})

    df['Housing'] = df['Housing'].map({'free': 0, 'rent': 1, 'own': 2})

    df['Saving accounts'] = df['Saving accounts'].map({
        'little': 0, 'moderate': 1, 'quite rich': 2, 'rich': 3
    })

    df['Checking account'] = df['Checking account'].map({
        'little': 0, 'moderate': 1, 'rich': 2
    })

    numeric_columns = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration']
    categorical_columns = ['Purpose']

    X_numeric = df[numeric_columns].values

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_categorical = encoder.fit_transform(df[categorical_columns])
 
    X = np.hstack((X_numeric, X_categorical))
    y = df['target'].values

    numeric_names = numeric_columns
    categorical_names = [f"{col}_{cat}" for col, cats in zip(categorical_columns, encoder.categories_) for cat in cats]
    all_feature_names = numeric_names + categorical_names

    X_df = pd.DataFrame(X, columns=all_feature_names)
    final_df = pd.concat([ pd.Series(y, name='target'),X_df], axis=1)

    return final_df

df = pd.read_excel('dataset.xlsx') #Ingresar tu dataset
prep_df = preprocess(df)

# Requiere que tengas credenciales AWS válidas (por ejemplo, usando `aws configure`)
# Asegúrate de tener sagemaker y boto3 instalados:
# pip install sagemaker boto3

endpoint_name = 'sagemaker-xgboost-2025-07-24-03-20-18-825'

#Configurar el predictor
predictor = Predictor(endpoint_name=endpoint_name, serializer=CSVSerializer())

X = prep_df.iloc[:, 1:].values  # Todas menos la primera columna
y = prep_df.iloc[:, 0].values   # Primera columna como target

#Invocar el endpoint
response = predictor.predict(X)
predictions = np.fromstring(response.decode('utf-8'), sep='\n')
binary_predictions = (predictions >= 0.5).astype(int)

accuracy = accuracy_score(y, binary_predictions)
precision = precision_score(y, binary_predictions)
recall = recall_score(y, binary_predictions)
f1 = f1_score(y, binary_predictions)
auc = roc_auc_score(y, predictions)
conf_matrix = confusion_matrix(y, binary_predictions)

#Metricas
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"AUC-ROC: {auc:.2f}")
print("Confusion Matrix:\n", conf_matrix)