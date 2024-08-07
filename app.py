from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('expense_model.joblib')

@app.route('/hello', methods=['POST'])
def hello():
    # Check for JSON content type
    if request.content_type != 'application/json':
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json()
    return jsonify({"message": "Hello, World!", "data": data})


@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON request data
    data = request.get_json()

    # Extract features from the request
    age = data['age']
    bmi = data['bmi']
    children = data['children']
    sex = data['sex']  # 0 for female, 1 for male
    smoker = data['smoker']  # 0 for non-smoker, 1 for smoker
    region = data['region']  # encoded region value

    # Create a feature array for prediction
    features = np.array([[age, bmi, children, sex, smoker, region]])

    # Predict charges
    prediction = model.predict(features)

    # Return the prediction as a JSON response
    return jsonify({'predicted_charges': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
