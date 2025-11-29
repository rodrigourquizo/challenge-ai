import pandas as pd
import boto3
import json
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from . import config

def extraer_target(descripcion):
    """Extracts target label from description."""
    if not isinstance(descripcion, str):
        return None
    if 'good risk' in descripcion.lower():
        return 'good risk'
    if 'buen riesgo' in descripcion.lower():
        return 'good risk'
    if 'bad risk' in descripcion.lower():
        return 'bad risk'
    if 'mal riesgo' in descripcion.lower():
        return 'bad risk'
    return None

def generate_descriptions(df, bedrock_client=None):
    """Generates descriptions using AWS Bedrock."""
    if bedrock_client is None:
        bedrock_client = boto3.client('bedrock-runtime', region_name=config.AWS_REGION)

    descriptions = []
    for i in range(len(df)):
        fila = df.iloc[i]
        
        ahorros = fila['Saving accounts'] if pd.notna(fila['Saving accounts']) else 'desconocido'
        cuenta_corriente = fila['Checking account'] if pd.notna(fila['Checking account']) else 'desconocido'
        mapa_trabajo = {0: 'no cualificado no residente', 1: 'no cualificado residente', 2: 'cualificado', 3: 'altamente cualificado'}
        desc_trabajo = mapa_trabajo.get(fila['Job'], 'desconocido')

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
        })
        try:
            response = bedrock_client.invoke_model(
                modelId=config.BEDROCK_MODEL_ID,
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

    return descriptions

def preprocess_data(df):
    """Preprocesses the dataframe for the model."""
    # Ensure description exists or generate it (skipping generation for now to keep it simple, assuming it might be done or we just need preprocessing for inference if descriptions are there)
    # Based on original code, descriptions are generated then used to create target.
    # But for inference on new data, do we need target? 
    # The original code 'preprocess' function did EVERYTHING: generation, labeling, encoding.
    
    # Let's split it. If 'description' is not in df, generate it.
    if 'description' not in df.columns:
        print("Generating descriptions...")
        df['description'] = generate_descriptions(df)
    
    # Labeling
    if 'target' not in df.columns:
        df['target'] = df['description'].apply(extraer_target)
    
    # Encoding
    df_processed = df.copy()
    df_processed['target'] = df_processed['target'].map({'good risk': 0, 'bad risk': 1})

    if df_processed['Sex'].dtype == 'object':
        df_processed['Sex'] = df_processed['Sex'].map({'female': 0, 'male': 1})

    df_processed['Housing'] = df_processed['Housing'].map({'free': 0, 'rent': 1, 'own': 2})

    df_processed['Saving accounts'] = df_processed['Saving accounts'].map({
        'little': 0, 'moderate': 1, 'quite rich': 2, 'rich': 3
    })

    df_processed['Checking account'] = df_processed['Checking account'].map({
        'little': 0, 'moderate': 1, 'rich': 2
    })
    
    # Handle NaNs if any (original code didn't explicitly handle them other than map, which turns unknown to NaN)
    # Assuming input is clean enough or map handles it.

    X_numeric = df_processed[config.NUMERIC_COLUMNS].values

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_categorical = encoder.fit_transform(df_processed[config.CATEGORICAL_COLUMNS])
 
    X = np.hstack((X_numeric, X_categorical))
    
    # If target is all NaN (e.g. inference without labels), we might just return X
    # But original code returns final_df with target.
    
    y = df_processed['target'].values

    numeric_names = config.NUMERIC_COLUMNS
    categorical_names = [f"{col}_{cat}" for col, cats in zip(config.CATEGORICAL_COLUMNS, encoder.categories_) for cat in cats]
    all_feature_names = numeric_names + categorical_names

    X_df = pd.DataFrame(X, columns=all_feature_names)
    final_df = pd.concat([pd.Series(y, name='target'), X_df], axis=1)

    return final_df
