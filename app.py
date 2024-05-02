from flask import Flask, request, render_template
import pickle
import xgboost as xgb
import numpy as np
from joblib import load


app = Flask(__name__)

# Load the XGBoost model
with open('xgboost.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the StandardScaler object
# with open('scaler_model.pkl', 'rb') as scaler_file:
#     scaler = pickle.load(scaler_file)
# print(type(scaler))  # This should output something like <class 'sklearn.preprocessing._data.StandardScaler'>


scaler = load('scaler_model.pkl')
print(type(scaler))
print("Features expected by the scaler:", scaler.feature_names_in_)

# # Print the attributes of the scaler
# print("Mean of the features:", scaler.mean_)
# print("Scale of the features (Standard Deviation):", scaler.scale_)

#loading gdp scalar
with open('gdp_model.pkl', 'rb') as gdp_file:
    sgdp = pickle.load(gdp_file)

# Define your list of countries and continents from your model's training data
country_continent_features = [
    'country_Argentina', 'country_Barbados', 'country_Bolivia', 'country_Brazil',
    'country_Canada', 'country_Chile', 'country_Colombia', 'country_Costa Rica',
    'country_Cuba', 'country_Dominica', 'country_Dominican Republic', 'country_Ecuador',
    'country_El Salvador', 'country_Guatemala', 'country_Haiti', 'country_Honduras',
    'country_Jamaica', 'country_Mexico', 'country_Nicaragua', 'country_Panama',
    'country_Paraguay', 'country_Peru', 'country_Puerto Rico', 'country_Saint Lucia',
    'country_Trinidad and Tobago', 'country_United States', 'country_Uruguay', 'country_Venezuela',
    'continent_North America', 'continent_South America'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    actual_prediction = None
    if request.method == 'POST':
        print(request.form)
        # Extract features from the form input
        numeric_features = [
            float(request.form.get('year', 0)),  # Adding the 'year' column
            float(request.form.get('population', 0)),
            float(request.form.get('coal_production', 0)),
            float(request.form.get('gas_production', 0)),
            float(request.form.get('low_carbon_electricity', 0)),
            float(request.form.get('nuclear_elec_per_capita', 0)),
            float(request.form.get('nuclear_electricity', 0)),
            float(request.form.get('oil_production', 0)),
            float(request.form.get('other_renewable_electricity', 0)),
            float(request.form.get('primary_energy_consumption', 0))
        ]

        # Initialize one-hot encoded features with zeros
        one_hot_features = [0.0] * len(country_continent_features)

        # Check and set one-hot features based on checkboxes
        for i, feature in enumerate(country_continent_features):
            if request.form.get(feature) == 'on':
                one_hot_features[i] = 1.0

        # Combine numeric and one-hot encoded features
        combined_features = numeric_features + one_hot_features

        # Convert the combined feature list to a NumPy array
        features_array = np.array([combined_features])

        # Scale the entire feature array since the scaler was trained on all features
        scaled_features = scaler.transform(features_array)

        # Predict using the XGBoost model
        prediction = model.predict(scaled_features)[0]
        actual_prediction = sgdp.inverse_transform([[prediction]])[0][0]

    return render_template('index.html', prediction=actual_prediction)

if __name__ == '__main__':
    app.run(debug=True)
