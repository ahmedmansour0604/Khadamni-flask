
import pickle
import joblib
import sys
import pandas as pd

def try_load_optimized(path):
    print(f"\n--- Inspecting {path} with latin1 ---")
    obj = None
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f, encoding='latin1')
        print("Loaded with pickle (latin1)")
    except Exception as e:
        print(f"Pickle (latin1) failed: {e}")
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            print("Loaded with pickle (default)")
        except Exception as e2:
            print(f"Pickle (default) failed: {e2}")
    
    if obj:
        print(f"Type: {type(obj)}")
        if hasattr(obj, 'n_features_in_'):
            print(f"n_features_in_: {obj.n_features_in_}")

try_load_optimized('knn_model.pkl')
try_load_optimized('scaler.pkl')
