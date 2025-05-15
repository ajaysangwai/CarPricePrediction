from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and data
model = pickle.load(open("LinearRegressionModel.pkl", 'rb'))
car = pd.read_csv("Cleaned_Car_data.csv")

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique()) 
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())
    return render_template('index.html',
                           companies=companies,
                           car_models=car_models,
                           years=years,
                           fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_model')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        kms_driven = int(request.form.get('kilo_driven'))

        # For debugging
        print(company, car_model, year, fuel_type, kms_driven)

        # Prepare DataFrame
        input_df = pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

        prediction = model.predict(input_df)[0]
        estimated_price = round(prediction, 2)

        return render_template('index.html',
                               companies=sorted(car['company'].unique()),
                               car_models=sorted(car['name'].unique()),
                               years=sorted(car['year'].unique(), reverse=True),
                               fuel_types=sorted(car['fuel_type'].unique()),
                               prediction_text=f"Estimated Price: ₹{estimated_price:,.2f}")

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
