from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression

api = Flask(__name__)

# Dummy data for demonstration
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 6, 8, 10])

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Endpoint to predict using the trained model
@api.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    x_value = data['x']
    x_value = np.array([[x_value]])
    prediction = model.predict(x_value)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
