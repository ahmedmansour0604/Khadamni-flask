
import pickle
import joblib
import pandas as pd
import sklearn
import sys

print(f"Python version: {sys.version}")
print(f"Sklearn version: {sklearn.__version__}")
print(f"Pandas version: {pd.__version__}")

def try_load(path):
    print(f"\n--- Intepecting {path} ---")
    obj = None
    # Try pickle
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print("Loaded with pickle")
    except Exception as e_pkl:
        print(f"Pickle load failed: {e_pkl}")
        # Try joblib
        try:
            obj = joblib.load(path)
            print("Loaded with joblib")
        except Exception as e_job:
            print(f"Joblib load failed: {e_job}")
    
    if obj is not None:
        print(f"Type: {type(obj)}")
        if hasattr(obj, 'n_features_in_'):
            print(f"n_features_in_: {obj.n_features_in_}")
        if hasattr(obj, 'feature_names_in_'):
            print(f"feature_names_in_: {obj.feature_names_in_}")
        if hasattr(obj, 'n_neighbors'):
            print(f"n_neighbors: {obj.n_neighbors}")

try_load('knn_model.pkl')
try_load('scaler.pkl')
