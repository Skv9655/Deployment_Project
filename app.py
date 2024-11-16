from flask import Flask, request, jsonify
import pickle
import numpy as np


app = Flask(__name__)

# Load the model
try:
    with open('C:/Users/skv96/Downloads/model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    raise Exception("Model file not found. Please ensure the path to'model.pkl' is correct.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('data')
        if data is None:
            return jsonify({'error': 'No data provided.'}), 400

        # Convert the data to a numpy array for prediction
        data = np.array(data)
        
        # Make prediction
        prediction = model.predict(data).tolist()

        return jsonify({'prediction': prediction})
    except ValueError as ve:
        return jsonify({'error': f'ValueError: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)