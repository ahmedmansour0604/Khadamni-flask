
import requests
import json

url = 'http://127.0.0.1:5000/recommend'
data = {
    'work_year': 2024,
    'employment_type': 'FT',
    'remote_ratio': 100,
    'job_title': 'Data Scientist',
    'experience_level': 'SE',
    'company_size': 'M',
    'employee_residence': 'US',
    'company_location': 'US',
    'salary_currency': 'USD',
    'salary_in_usd': 150000
}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, json=data, timeout=5)
    print(f"Status Code: {response.status_code}")
    print("Response:", response.text)
except Exception as e:
    print(f"Request failed: {e}")
