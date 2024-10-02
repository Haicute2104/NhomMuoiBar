from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the saved model (replace 'best_model.pkl' with the actual best model if needed)
model = joblib.load('best_model.pkl')  # Use the best model
model2 = joblib.load('best_model_lasso.pkl')
model3 = joblib.load('best_model_MLP.pkl')
model4 = joblib.load('best_model_Stacking.pkl')

# Load the weather encoder to ensure we match the user selection
weather_data = pd.read_csv('Weather Data.csv')
label_encoder = joblib.load('label_encoder.pkl')  # Load the saved label encoder for Weather

# Available weather conditions
weather_conditions = list(weather_data['Weather'].unique())

@app.route('/')
def home():
    return render_template('index.html', weather_conditions=weather_conditions)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        dew_point = float(request.form['dew_point'])
        humidity = float(request.form['humidity'])
        wind_speed = float(request.form['wind_speed'])
        visibility = float(request.form['visibility'])
        pressure = float(request.form['pressure'])
        weather = request.form['weather']


        # Encode the weather condition
        weather_encoded = label_encoder.transform([weather])[0]

        # Create feature array for prediction
        features = np.array([[ dew_point, humidity, wind_speed, visibility, pressure, weather_encoded]])

        # Predict using the loaded model
        prediction = model.predict(features)

        prediction2 = model2.predict(features)

        prediction3 = model3.predict(features)

        prediction4 = model4.predict(features)
        # Return the prediction result
        return render_template('index.html', prediction=prediction[0], prediction2 = prediction2[0], prediction3 = prediction3[0],prediction4 = prediction4[0], weather_conditions=weather_conditions)

if __name__ == '__main__':
    app.run(debug=True)
