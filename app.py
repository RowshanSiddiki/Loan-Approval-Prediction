from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("models/loan_model.pkl")
education_encoder = joblib.load("models/education_encoder.pkl")
employment_encoder = joblib.load("models/employment_encoder.pkl")
feature_names = joblib.load("models/feature_names.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")


@app.route('/')
def home():
    return render_template("index.html")


# Predict 
@app.route("/predict", methods=["POST"]) 
def predict():
    no_of_dependents = int(request.form['no_of_dependents'])
    education = request.form['education']
    education = education_encoder.transform([education])[0]
    
    self_employed = request.form['self_employed']
    self_employed = employment_encoder.transform([self_employed])[0]
    income_annum = float(request.form['income_annum'])
    
    loan_amount = float(request.form['loan_amount'])
    loan_term = int(request.form['loan_term'])
    cibil_score = int(request.form['cibil_score'])
    
    residential_assets_value = float(request.form['residential_assets_value'])
    commercial_assets_value = float(request.form['commercial_assets_value'])
    luxury_assents_value = float(request.form['luxury_assets_value'])
    bank_asset_value = float(request.form['bank_asset_value'])

    data = pd.DataFrame([{
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assents_value,
        "bank_asset_value": bank_asset_value
    }])
    
    
    # Rearrange the feature names in the same order as the model was trained
    data = data[feature_names]
    
    prediction = model.predict(data)
    result = target_encoder.inverse_transform(prediction)[0]
    
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)