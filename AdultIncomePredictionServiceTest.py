import pytest
from app import app

@pytest.fixture
def client():
    # Set up a test client for the Flask app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_homepage_loads(client):
    """Test if the homepage loads successfully"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Income Prediction' in response.data

def test_prediction_success(client):
    """Test the prediction endpoint with valid sample input"""
    sample_input = {
        'age': '35',
        'workclass': 'Private',
        'fnlwgt': '200000',
        'education': 'Bachelors',
        'education.num': '13',
        'marital.status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'native.country': 'United-States'
    }

    response = client.post('/predict', data=sample_input, follow_redirects=True)
    assert response.status_code == 200
    assert b'Income' in response.data  # Check if prediction result is rendered

def test_prediction_missing_fields(client):
    """Test the prediction with missing required fields"""
    incomplete_input = {
        'age': '30',
        'workclass': '',
        'fnlwgt': '180000',
        'education': 'HS-grad',
        'education.num': '9',
        'marital.status': 'Never-married',
        'occupation': '',
        'relationship': 'Not-in-family',
        'race': 'Black',
        'sex': 'Female',
        'native.country': ''
    }

    response = client.post('/predict', data=incomplete_input, follow_redirects=True)
    assert response.status_code == 200
    assert b'Income' in response.data  # Even with missing values, it should handle gracefully

def test_invalid_method(client):
    """Test if invalid methods are handled properly"""
    response = client.get('/predict')
    assert response.status_code in (405, 400)  # 405 Method Not Allowed or 400 Bad Request
