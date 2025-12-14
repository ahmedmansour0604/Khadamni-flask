
import pytest
import json
import sys
import os

# Ensure current dir is in path
sys.path.append(os.getcwd())

from app import app
# Manually trigger load in the test setup instead of importing if imports fail due to contexts

@pytest.fixture
def client():
    app.config['TESTING'] = True
    # Trigger loading via a special route or just by ensuring the module is loaded
    # app.py's if __name__ == main calls the load but importing doesn't auto call it.
    # We can call the functions if we can import them.
    from app import load_resources
    load_resources()
    
    with app.test_client() as client:
        yield client

def test_recommendation_flow(client):
    payload = {
        "work_year": "2025",
        "employment_type": "FT",
        "remote_ratio": "0",
        "job_title": "Data Scientist",
        "experience_level": "SE",
        "company_size": "M",
        "company_location": "US",
        "employee_residence": "US",
        "salary_in_usd": "150000"
    }
    
    response = client.post('/recommend', json=payload)
    data = response.get_json()
    
    print("\n--- Response Code ---")
    print(response.status_code)
    
    print("\n--- Response Data ---")
    print(json.dumps(data, indent=2))
    
    assert response.status_code == 200 or data.get('status') == 'error'
    
    # Test 2: Data Engineer
    payload['job_title'] = "Data Engineer"
    response = client.post('/recommend', json=payload)
    print("\n--- Response Data Engineer ---")
    data = response.get_json()
    print(json.dumps(data, indent=2))
    assert response.status_code == 200

if __name__ == "__main__":
    # Manually running if pytest not installed
    app.config['TESTING'] = True
    load_data()
    load_models()
    with app.test_client() as client:
        test_recommendation_flow(client)
