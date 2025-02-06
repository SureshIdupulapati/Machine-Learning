from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open(r'Loan_Aooroval_Prediction.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        Credit_History = float(request.form['Credit_History'])
        Property_Area = float(request.form['Property_Area'])
        Income = float(request.form['Income'])

        # Prepare data for prediction
        data = np.array([[Credit_History, Property_Area, Income]])

        # Make prediction
        prediction = model.predict(data)[0]
        result = "Loan Approved" if prediction == 1 else "Loan Rejected"

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=False)
