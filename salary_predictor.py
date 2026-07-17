"""
SalarySage - Advanced Salary Prediction using Ensemble Learning
------------------------------------------------------------------
Dataset: Data Science Job Salaries (Kaggle) - ds_salaries.csv
Models Compared: Random Forest, Gradient Boosting, XGBoost, Voting Ensemble
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost nahi mila, pip install xgboost karo. Filhaal isko skip kar rahe hain.")

# ==================== STEP 1: Load Dataset ====================
df = pd.read_csv("ds_salaries.csv")
print("----- Dataset Shape -----")
print(df.shape)
print("\n----- First 5 rows -----")
print(df.head())
print("\n----- Column Info -----")
print(df.info())

# ==================== STEP 2: EDA (Exploratory Data Analysis) ====================
print("\n----- Missing Values -----")
print(df.isnull().sum())

plt.figure(figsize=(8, 5))
sns.histplot(df['salary_in_usd'], bins=40, kde=True, color='teal')
plt.title("Salary Distribution (USD)")
plt.savefig("eda_salary_distribution.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(x='experience_level', y='salary_in_usd', data=df, order=['EN','MI','SE','EX'])
plt.title("Salary by Experience Level")
plt.savefig("eda_salary_by_experience.png")
plt.close()

# ==================== STEP 3: Feature Selection & Cleaning ====================
# Useful columns choose kar rahe hain (job_title aur locations mein bohot unique values hain,
# isliye unko simplify karenge taaki model overfit na ho)

# Top 10 job titles rakho, baaki ko "Other" bana do
top_titles = df['job_title'].value_counts().nlargest(10).index
df['job_title_grouped'] = df['job_title'].apply(lambda x: x if x in top_titles else 'Other')

# Company location bhi simplify - top 10 countries, baaki "Other"
top_locations = df['company_location'].value_counts().nlargest(10).index
df['company_location_grouped'] = df['company_location'].apply(lambda x: x if x in top_locations else 'Other')

features = ['work_year', 'experience_level', 'employment_type', 'job_title_grouped',
            'remote_ratio', 'company_location_grouped', 'company_size']
target = 'salary_in_usd'

model_df = df[features + [target]].copy()

# ==================== STEP 4: Encoding ====================
categorical_cols = ['experience_level', 'employment_type', 'job_title_grouped',
                     'company_location_grouped', 'company_size']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col])
    label_encoders[col] = le  # baad mein naye input decode karne ke liye save kar rahe hain

X = model_df.drop(target, axis=1)
y = model_df[target]

# ==================== STEP 5: Train-Test Split ====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining size: {X_train.shape}, Testing size: {X_test.shape}")

# ==================== STEP 6: Multiple Models Train karo ====================
models = {
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
}

if XGB_AVAILABLE:
    models["XGBoost"] = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)

results = []

for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    preds = mdl.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    print(f"\n{name} -> MAE: {mae:,.2f} | RMSE: {rmse:,.2f} | R2: {r2:.4f}")

# ==================== STEP 7: Voting Ensemble (Sab models ko combine karna) ====================
estimators = [(name, mdl) for name, mdl in models.items()]
voting_model = VotingRegressor(estimators=estimators)
voting_model.fit(X_train, y_train)
voting_preds = voting_model.predict(X_test)

mae_v = mean_absolute_error(y_test, voting_preds)
rmse_v = np.sqrt(mean_squared_error(y_test, voting_preds))
r2_v = r2_score(y_test, voting_preds)
results.append({"Model": "Voting Ensemble (All Combined)", "MAE": mae_v, "RMSE": rmse_v, "R2": r2_v})
print(f"\nVoting Ensemble -> MAE: {mae_v:,.2f} | RMSE: {rmse_v:,.2f} | R2: {r2_v:.4f}")

# ==================== STEP 8: Results Comparison Table ====================
results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
print("\n----- Model Comparison -----")
print(results_df.to_string(index=False))

plt.figure(figsize=(8, 5))
sns.barplot(x='R2', y='Model', data=results_df, hue='Model', legend=False, palette='mako')
plt.title("Model Comparison (R2 Score)")
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()

# ==================== STEP 9: Hyperparameter Tuning (Best model pe) ====================
print("\n----- Hyperparameter Tuning: Random Forest -----")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [8, 10, 15],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid,
                            cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Params:", grid_search.best_params_)
best_model = grid_search.best_estimator_

final_preds = best_model.predict(X_test)
print(f"Tuned Model R2: {r2_score(y_test, final_preds):.4f}")

# ==================== STEP 10: Feature Importance ====================
importances = best_model.feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feat_df, hue='Feature', legend=False, palette='viridis')
plt.title("Feature Importance (Tuned Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

print("\n----- Feature Importance -----")
print(feat_df)

# ==================== STEP 11: Save Model ====================
joblib.dump(best_model, "salary_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
print("\nModel saved as salary_model.pkl")

# ==================== STEP 12: Custom Prediction Function ====================
def predict_salary(work_year, experience_level, employment_type, job_title, remote_ratio, company_location, company_size):
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

    input_df = input_df[X.columns]
    prediction = best_model.predict(input_df)
    return prediction[0]

# Example prediction
sample_salary = predict_salary(
    work_year=2023, experience_level='SE', employment_type='FT',
    job_title='Data Scientist', remote_ratio=100,
    company_location='US', company_size='M'
)
print(f"\nSample Predicted Salary: ${sample_salary:,.2f}")
# ==================== STEP 13: Interactive Prediction (User Input) ====================
def interactive_predict():
    print("\n===== Salary Prediction Tool =====")
    print("Apni details daalo, salary predict ho jayegi:\n")

    print("Experience Levels: EN (Entry), MI (Mid), SE (Senior), EX (Executive)")
    exp = input("Experience Level: ").strip().upper()

    print("\nEmployment Types: FT (Full-time), PT (Part-time), CT (Contract), FL (Freelance)")
    emp_type = input("Employment Type: ").strip().upper()

    print(f"\nCommon Job Titles: {list(top_titles)}")
    job = input("Job Title: ").strip()

    remote = input("\nRemote Ratio (0 = Office, 50 = Hybrid, 100 = Fully Remote): ").strip()

    print(f"\nCommon Company Locations (country code): {list(top_locations)}")
    location = input("Company Location: ").strip().upper()

    print("\nCompany Size: S (Small), M (Medium), L (Large)")
    size = input("Company Size: ").strip().upper()

    year = input("\nWork Year (e.g. 2023): ").strip()

    try:
        predicted = predict_salary(
            work_year=int(year),
            experience_level=exp,
            employment_type=emp_type,
            job_title=job,
            remote_ratio=int(remote),
            company_location=location,
            company_size=size
        )
        print(f"\n✅ Predicted Salary: ${predicted:,.2f} per year")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Check karo values sahi format mein daali hain (jaise numbers ke liye numbers).")
