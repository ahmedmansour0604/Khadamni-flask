
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')

# --- Configuration & Global Variables ---
DATA_PATH = 'salaries.csv'
KNN_MODEL_PATH = 'knn_model.pkl'
SCALER_PATH = 'scaler.pkl'

df = None
knn_model = None
scaler = None

# Mappings (Standard/User Defined)
EXPERIENCE_LEVEL_MAP = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
COMPANY_SIZE_MAP = {'S': 0, 'M': 1, 'L': 2}
EMPLOYMENT_TYPE_MAP = {'CT': 0, 'FL': 1, 'FT': 2, 'PT': 3}

# Job Titles Grouping (from User)
JOB_TITLE_GROUPS = [
    'Other', 'Data Scientist', 'Engineer', 'AI Engineer',
    'Business Intelligence Engineer', 'Manager', 'Product Manager',
    'Data Engineer', 'Machine Learning Engineer', 'Software Engineer',
    'Data Analyst', 'Research Scientist', 'Developer', 'Associate', 'Analyst',
    'Analytics Engineer', 'Systems Engineer', 'Director', 'Consultant',
    'Solutions Architect', 'Data Manager', 'Applied Scientist', 'Data Architect',
    'Data Specialist', 'Business Intelligence Analyst', 'Architect',
    'Engineering Manager', 'Platform Engineer', 'Research Engineer',
    'Software Developer', 'Software Development Engineer'
]

# Frequency Maps
frequency_maps = {
    'employee_residence': {},
    'company_location': {},
    'salary_currency': {}
}

# --- Helper Functions ---

def get_job_title_group(title):
    if title in JOB_TITLE_GROUPS:
        return title
    for group in JOB_TITLE_GROUPS:
        if group != 'Other' and group.lower() in title.lower():
            return group
    return 'Other'


def train_new_model():
    """Retrain model from scratch using current data and encodings."""
    global knn_model, knn_cosine_model, scaler, df, frequency_maps
    print("Retraining models from scratch...")
    
    if df is None:
        print("Error: No data to train on.")
        return

    # Create a training copy
    train_df = df.copy()
    
    # 1. Map Ordinals/Labels
    train_df['experience_level_ord'] = train_df['experience_level'].map(EXPERIENCE_LEVEL_MAP).fillna(0)
    train_df['company_size_ord'] = train_df['company_size'].map(COMPANY_SIZE_MAP).fillna(1)
    train_df['employment_type_encoded'] = train_df['employment_type'].map(EMPLOYMENT_TYPE_MAP).fillna(2)
    
    # 2. Map Job Titles
    sorted_groups = sorted(JOB_TITLE_GROUPS)
    job_map = {name: i for i, name in enumerate(sorted_groups)}
    
    def encode_job(title):
        group = get_job_title_group(title)
        return job_map.get(group, job_map.get('Other'))
        
    train_df['job_title_grouped_encoded'] = train_df['job_title'].apply(encode_job)
    
    # 3. Frequency Encoding
    for col in frequency_maps:
        freq_map = train_df[col].value_counts(normalize=True).to_dict()
        frequency_maps[col] = freq_map 
        train_df[f'{col}_freq'] = train_df[col].map(freq_map).fillna(0)
        
    # 4. Prepare Features
    feature_cols = [
        'work_year', 'employment_type_encoded', 'remote_ratio', 
        'job_title_grouped_encoded', 'experience_level_ord', 'company_size_ord',
        'employee_residence_freq', 'company_location_freq', 'salary_currency_freq', 
        'salary_in_usd'
    ]
    
    X = train_df[feature_cols].values
    
    # 5. Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 6. Train KNN (Minkowski)
    knn_model = NearestNeighbors(n_neighbors=5, metric='minkowski')
    knn_model.fit(X_scaled)

    # 7. Train KNN (Cosine)
    knn_cosine_model = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn_cosine_model.fit(X_scaled)
    
    print("Models retrained successfully.")

def load_resources():
    global df, knn_model, knn_cosine_model, scaler, frequency_maps
    
    # Load Data
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print("Data loaded.")
    else:
        print("CRITICAL ERROR: Data not found.")
        return

    # Compute Frequencies Initial
    for col in frequency_maps:
        frequency_maps[col] = df[col].value_counts(normalize=True).to_dict()

    # Try Loading Models
    loaded = False
    try:
        # Define path for cosine model if not globally defined (assuming same dir)
        KNN_COSINE_PATH = 'knn_cosine_model.pkl'

        if os.path.exists(KNN_MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(KNN_COSINE_PATH):
            try:
                with open(KNN_MODEL_PATH, 'rb') as f:
                    knn_model = pickle.load(f)
                with open(KNN_COSINE_PATH, 'rb') as f:
                    knn_cosine_model = pickle.load(f)
                with open(SCALER_PATH, 'rb') as f:
                    scaler = pickle.load(f)
                
                if scaler.n_features_in_ != 10:
                    raise ValueError(f"Scaler expects {scaler.n_features_in_} features, but code provides 10.")

                loaded = True
                print("Models loaded via pickle.")
            except:
                print("Pickle load failed, trying joblib...")
                knn_model = joblib.load(KNN_MODEL_PATH)
                knn_cosine_model = joblib.load(KNN_COSINE_PATH)
                scaler = joblib.load(SCALER_PATH)
                
                if scaler.n_features_in_ != 10:
                    raise ValueError(f"Scaler expects {scaler.n_features_in_} features, but code provides 10.")
                
                loaded = True
                print("Models loaded via joblib.")
    except Exception as e:
        print(f"Model loading failed: {e}")
    
    if not loaded:
        print("Could not load models (or mismatch found). Retraining...")
        train_new_model()

# --- Routes ---

@app.route('/')
@app.route('/job_recommendation.html')
def home():
    return render_template('job_recommendation.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        if knn_model is None or scaler is None:
             return jsonify({'status': 'error', 'message': 'Model not available'}), 503

        data = request.json
        
        # Inputs
        work_year = int(data.get('work_year', 2025))
        emp_type = data.get('employment_type', 'FT')
        remote = float(data.get('remote_ratio', 0))
        job = data.get('job_title', 'Data Scientist')
        exp = data.get('experience_level', 'SE')
        size = data.get('company_size', 'M')
        res = data.get('employee_residence', 'US')
        loc = data.get('company_location', 'US')
        curr = data.get('salary_currency', 'USD')
        salary = float(data.get('salary_in_usd', 100000))

        # Encode
        emp_val = EMPLOYMENT_TYPE_MAP.get(emp_type, 2)
        
        sorted_jobs = sorted(JOB_TITLE_GROUPS)
        job_map = {name: i for i, name in enumerate(sorted_jobs)}
        job_group = get_job_title_group(job)
        job_val = job_map.get(job_group, job_map.get('Other'))
        
        exp_val = EXPERIENCE_LEVEL_MAP.get(exp, 2)
        size_val = COMPANY_SIZE_MAP.get(size, 1)
        
        res_freq = frequency_maps['employee_residence'].get(res, 0)
        loc_freq = frequency_maps['company_location'].get(loc, 0)
        curr_freq = frequency_maps['salary_currency'].get(curr, 0) # Fallback if unknown
        if curr_freq == 0 and 'USD' in frequency_maps['salary_currency']:
             curr_freq = frequency_maps['salary_currency']['USD'] # Reasonable default ?

        # Vector: work_year, employment_type, remote_ratio, job_title, exp, size, res_freq, loc_freq, curr_freq, salary
        features = np.array([[
            work_year, emp_val, remote, job_val, exp_val, size_val, 
            res_freq, loc_freq, curr_freq, salary
        ]])
        
        # Scale
        features_scaled = scaler.transform(features)
        
        # Predict
        distances, indices = knn_model.kneighbors(features_scaled, n_neighbors=10)
        distances_cosine, indices_cosine = knn_cosine_model.kneighbors(features_scaled, n_neighbors=20)
        
        recommendations = []
        seen_indices = set()
        seen_signatures = set()
        
        # Helper to process indices
        def add_recs(inds, limit=3):
            count = 0
            for idx in inds[0]:
                if count >= limit:
                    break
                
                # Check index uniqueness
                if idx in seen_indices or idx >= len(df):
                    continue
                    
                row = df.iloc[idx].to_dict()
                
                # Check strict content uniqueness
                # Include ALL visible fields
                sig = (
                    row.get('job_title'), 
                    row.get('company_location'), 
                    row.get('work_year'),
                    row.get('salary_in_usd'),
                    row.get('experience_level'),
                    row.get('employment_type'),
                    row.get('company_size'),
                    row.get('remote_ratio')
                )
                
                if sig in seen_signatures:
                    continue
                    
                seen_indices.add(idx)
                seen_signatures.add(sig)
                
                clean_row = {k: (v if pd.notna(v) else "") for k, v in row.items()}
                recommendations.append(clean_row)
                count += 1

        add_recs(indices, limit=3)
        add_recs(indices_cosine, limit=3)
        
        return jsonify({'status': 'success', 'recommendations': recommendations})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    load_resources()
    app.run(debug=True, port=5000)
