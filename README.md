💰 SalarySage - Salary Prediction using Ensemble Learning

SalarySage is a machine learning web application that predicts Data Science job salaries using an ensemble of Random Forest, Gradient Boosting, and XGBoost models, trained on real-world global salary data (2020–2025). It includes a full interactive web interface built with Flask, model insight visualizations, and multi-currency support.


📌 IBM PBEL Internship Project



🔗 Live Demo: https://salarysage.onrender.com

🔗 GitHub Repository: https://github.com/unnatissrivastava/SalarySage

📊 Presentation: https://1drv.ms/p/c/20e98d3c7ba34c81/IQBc_rcItNK7RJax4LkzZRy9AWdDjg3DQiLpS5JmFtlY7kU?e=pdmv3k&nav=eyJzSWQiOjI1Nn0

⚠️ Note: The live demo is hosted on Render's free tier. If it hasn't been visited in a while, the first load may take 30–50 seconds to "wake up."




📋 Table of Contents


Overview
Features
Project Structure
How to Run Locally
Machine Learning Approach
Model Performance
Tech Stack
Future Improvements
Author



📖 Overview

This project was built to answer a simple question: "What should a Data Science professional expect to earn, based on their experience, role, and location?"

Rather than relying on guesswork or scattered salary forums, SalarySage uses a trained ensemble machine learning model built on real, publicly available salary data to generate consistent, data-backed salary estimates through a clean, interactive web interface.


✨ Features


🔮 Interactive Salary Predictor enter your experience level, job title, employment type, remote ratio, company location, company size, and work year to get an instant salary estimate
🌍 All Countries Supported : the location dropdown includes every country present in the dataset, not just the most common ones
🤖 Ensemble Learning : combines Random Forest, Gradient Boosting, and XGBoost via a Voting Regressor for more robust predictions than any single model
🎯 Hyperparameter Tuning : uses GridSearchCV to find the best-performing model configuration
💱 Multi-Currency Support — view predictions in USD, INR, EUR, or GBP
📈 Career Paths Page — an interactive timeline showing how salary and skills evolve from Entry to Executive level
📊 Model Insights Page — visualizes model comparison, feature importance, and salary distribution trends
🌐 Fully Deployed — live, publicly accessible web app (not just a local script), auto-deployed via Render on every GitHub push



## 🗂️ Project Structure

```
SalarySage/
├── app.py                 # Flask web application (routes & prediction logic)
├── salary_predictor.py    # ML training script — trains & saves the model
├── ds_salaries.csv        # Dataset (2020–2025 Data Science job salaries)
├── requirements.txt       # Python dependencies
├── Procfile                # Deployment config for Render
├── salary_model.pkl        # Saved, trained best-performing model
├── label_encoders.pkl      # Saved label encoders for categorical features
├── README.md                # Project documentation (this file)
│
├── templates/                # HTML pages (Jinja2 templates)
│   ├── home.html             # Landing page
│   ├── predict.html          # Prediction input form
│   ├── result.html           # Prediction result page
│   ├── insights.html         # Model insights & graphs page
│   ├── careers.html          # Interactive career growth timeline
│   └── contact.html          # Contact/about page
│
└── static/
    ├── style.css              # Site styling
    └── *.png                  # Graphs (EDA, model comparison, etc.)
```

⚙️ How to Run This Project Locally

1. Clone the repository

bashgit clone https://github.com/unnatissrivastava/SalarySage.git
cd SalarySage

2. Install dependencies

It's recommended to use a virtual environment:

bashpython -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

Then install requirements:

bashpip install -r requirements.txt

3. (Optional) Retrain the model

The trained model (salary_model.pkl) is already included, but you can retrain it from scratch:

bashpython salary_predictor.py

This will:


Load and clean ds_salaries.csv
Train Random Forest, Gradient Boosting, and XGBoost models
Combine them into a Voting Ensemble
Run hyperparameter tuning via GridSearchCV
Save the best model and generate evaluation graphs


4. Run the web application

bashpython app.py

5. Open in your browser

http://127.0.0.1:5000


🧠 Machine Learning Approach

StepDescriptionDatasetData Science Job Salaries — 2020 to 2025Target Variablesalary_in_usdFeatures UsedWork year, experience level, employment type, job title, remote ratio, company location, company sizePreprocessingLabel Encoding for categorical variables; top-10 grouping for high-cardinality fields (job title); all countries retained for locationModels TrainedRandom Forest Regressor, Gradient Boosting Regressor, XGBoost RegressorEnsemble MethodVoting Regressor (averages predictions across all three models)TuningGridSearchCV with 3-fold cross-validation, optimized for R² scoreEvaluation MetricsMean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² Score


📈 Model Performance

The Voting Ensemble was compared against each individual model on a held-out test set (20% of the data). The best-performing configuration — a tuned Random Forest — was selected as the final production model based on R² score.

Detailed comparison charts (model performance, feature importance, and salary distribution) are available on the Insights page of the app.


🛠️ Tech Stack

Machine Learning


Python 3
scikit-learn (Random Forest, Gradient Boosting, GridSearchCV)
XGBoost
pandas, numpy
joblib (model serialization)


Web Application


Flask (backend & routing)
HTML5, CSS3 (custom "Bitcoin DeFi"-inspired dark UI)
Jinja2 templating


Deployment


Render (web hosting, auto-deploy on push)
Gunicorn (production WSGI server)
Git & GitHub (version control)



🚀 Future Improvements


Expand the dataset to include non-tech job roles for broader salary predictions
Add authentication so users can save past predictions
Deploy on a paid tier or alternative host to eliminate the free-tier "sleep" delay
Add a REST API endpoint for programmatic access to predictions



👤 Author

Unnati Srivastava
GitHub: @unnatissrivastava


📄 License

This project was built for educational purposes as part of an IBM PBEL internship assignment on Ensemble Learning.
