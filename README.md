SalarySage 💰

A machine learning-powered web app that predicts salaries based on job title, remote work ratio, company location, and company size — with results converted into multiple currencies.

🔍 Overview

SalarySage uses a trained ML model to estimate salaries from real-world job market data. Users input their job details through a simple web interface, and the app returns a predicted salary in their chosen currency.

✨ Features


Predicts salary based on:

Job Title
Remote Work Ratio
Company Location
Company Size



Converts predicted salary into multiple currencies
Clean, simple web interface (Home, Predict, Result, Contact pages)
Exploratory Data Analysis (EDA) included — salary distribution & salary by experience
Model comparison and feature importance visualizations


🛠️ Tech Stack


Backend: Python, Flask
ML/Data: Pandas, Scikit-learn, Label Encoding
Frontend: HTML, CSS
Model Storage: Pickle (.pkl)


📁 Project Structure

SalarySage/
├── static/
│   └── style.css
├── templates/
│   ├── home.html
│   ├── predict.html
│   ├── result.html
│   └── contact.html
├── app.py                     # Flask app & prediction logic
├── salary_predictor.py        # Model training script
├── ds_salaries.csv            # Dataset
├── label_encoders.pkl         # Saved label encoders
├── salary_model.pkl           # Trained ML model
├── eda_salary_by_experience.png
├── eda_salary_distribution.png
├── feature_importance.png
├── model_comparison.png
└── requirements.txt

🚀 How to Run


Clone the repository


   git clone <your-repo-link>
   cd SalarySage


Install dependencies


   pip install -r requirements.txt


Run the app


   python app.py


Open your browser at http://localhost:5000


📊 Model

The model was trained on salary data (ds_salaries.csv) using categorical features encoded via LabelEncoder. Multiple models were compared (see model_comparison.png) and the best-performing one was saved as salary_model.pkl, along with feature importance analysis (feature_importance.png).

📌 Status

🔄 Work in progress part of my AI/ML learning journey.

🙋‍♀️ Author

Built while learning Data Science & ML — feedback welcome!