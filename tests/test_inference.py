import pytest
import pandas as pd
import numpy as np
from src import processing

def test_extraer_target():
    assert processing.extraer_target("This is a good risk profile") == "good risk"
    assert processing.extraer_target("High probability of default, bad risk") == "bad risk"
    assert processing.extraer_target("Unclear description") is None

def test_preprocess_structure():
    # Create a dummy dataframe
    data = {
        'Age': [30, 40],
        'Sex': ['male', 'female'],
        'Job': [2, 3],
        'Housing': ['own', 'rent'],
        'Saving accounts': ['little', 'rich'],
        'Checking account': ['moderate', 'little'],
        'Credit amount': [1000, 5000],
        'Duration': [12, 24],
        'Purpose': ['car', 'education'],
        'description': ['good risk', 'bad risk'] # Mock descriptions
    }
    df = pd.DataFrame(data)
    
    processed_df = processing.preprocess_data(df)
    
    assert 'target' in processed_df.columns
    assert processed_df.shape[1] > 10 # Should have encoded columns
    assert not processed_df.isnull().values.any()
