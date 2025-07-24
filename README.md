# GDP Prediction Web App

A web application for predicting the GDP of American countries using energy and population data, powered by machine learning (XGBoost). The app provides an interactive form for users to input features and select countries/continents, and returns a GDP prediction.

[Live Demo](https://gdp-prediction-app.onrender.com/)

## Features
- Predict GDP for American countries based on energy production/consumption and population data
- Interactive web form with predefined example datasets
- Country and continent selection (one-hot encoded)
- Visualization of countries and continents on a map
- Built with Flask and XGBoost

## Input Features
- Year
- Population
- Coal Production
- Gas Production
- Low Carbon Electricity
- Nuclear Electricity Per Capita
- Nuclear Electricity
- Oil Production
- Other Renewable Electricity
- Primary Energy Consumption
- Country (select one)
- Continent (North America, South America)

## Getting Started

### Prerequisites
- Python 3.7+

### Installation
1. Clone the repository or download the source code.
2. Navigate to the web app directory:
   ```bash
   cd src/phase3/Website
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App Locally
1. Ensure the following files are present in the directory:
   - `app.py`
   - `xgboost.pkl` (trained XGBoost model)
   - `scaler_model.pkl` (scikit-learn StandardScaler)
   - `gdp_model.pkl` (GDP scaler for inverse transformation)
   - `templates/index.html`
2. Start the Flask app:
   ```bash
   python app.py
   ```
3. Open your browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

## Usage
- Fill in the form fields with your data or use one of the predefined sets.
- Select the country and continent.
- Click **Predict** to get the GDP prediction.
- The result will be displayed on the page.

## File Structure
```
src/phase3/Website/
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── xgboost.pkl           # Trained XGBoost model
├── scaler_model.pkl      # StandardScaler for input features
├── gdp_model.pkl         # Scaler for GDP output
└── templates/
    └── index.html        # Web interface template
```

## Model Training
- The provided `.pkl` files are pre-trained and required to run the app.
- For model training and data preparation, see the Jupyter notebooks in `src/phase1/` and `src/phase2/`.

## Credits
- Developed as part of a Data Intelligence Capstone project.
- Built with [Flask](https://flask.palletsprojects.com/), [XGBoost](https://xgboost.readthedocs.io/), and [scikit-learn](https://scikit-learn.org/).

## License
This project is for educational purposes. 
