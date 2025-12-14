
import pickle
import numpy as np
import pandas as pd
import sklearn

print(f"Sklearn version: {sklearn.__version__}")

def inspect_pkl(path):
    print(f"\n--- Inspecting {path} ---")
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f"Type: {type(obj)}")
        
        if hasattr(obj, 'n_features_in_'):
            print(f"n_features_in_: {obj.n_features_in_}")
        if hasattr(obj, 'feature_names_in_'):
            print(f"feature_names_in_: {obj.feature_names_in_}")
        
    except Exception as e:
        print(f"Error loading {path}: {e}")

inspect_pkl('knn_model.pkl')
inspect_pkl('knn_cosine_model.pkl')
inspect_pkl('scaler.pkl')
