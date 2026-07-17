from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

best_model = joblib.load("salary_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

df = pd.read_csv("ds_salaries.csv")
top_titles = df['job_title'].value_counts().nlargest(10).index.tolist()
top_locations = sorted(df['company_location'].unique().tolist())

categorical_cols = ['experience_level', 'employment_type', 'job_title_grouped',
                     'company_location_grouped', 'company_size']
feature_order = ['work_year', 'experience_level', 'employment_type', 'job_title_grouped',
                  'remote_ratio', 'company_location_grouped', 'company_size']

CURRENCY_RATES = {
    'USD': 1,
    'INR': 83.5,
    'EUR': 0.92,
    'GBP': 0.79,
}

CURRENCY_SYMBOLS = {
    'USD': '$',
    'INR': '₹',
    'EUR': '€',
    'GBP': '£',
}


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        work_year = int(request.form['work_year'])
        experience_level = request.form['experience_level']
        employment_type = request.form['employment_type']
        job_title = request.form['job_title']
        remote_ratio = int(request.form['remote_ratio'])
        company_location = request.form['company_location']
        company_size = request.form['company_size']
        currency = request.form['currency']

        input_df = pd.DataFrame([{
            'work_year': work_year,
            'experience_level': experience_level,
            'employment_type': employment_type,
            'job_title_grouped': job_title if job_title in top_titles else 'Other',
            'remote_ratio': remote_ratio,
            'company_location_grouped': company_location if company_location in top_locations else 'Other',
            'company_size': company_size
        }])

        for col in categorical_cols:
            le = label_encoders[col]
            input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            input_df[col] = le.transform(input_df[col])

        input_df = input_df[feature_order]
        prediction_usd = best_model.predict(input_df)[0]

        converted_salary = prediction_usd * CURRENCY_RATES[currency]
        symbol = CURRENCY_SYMBOLS[currency]

        return render_template('result.html', salary=f"{converted_salary:,.2f}", symbol=symbol, currency=currency)

    return render_template('predict.html', titles=top_titles, locations=top_locations)


@app.route('/insights')
def insights():
    return render_template('insights.html')


@app.route('/careers')
def careers():
    return render_template('careers.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True)