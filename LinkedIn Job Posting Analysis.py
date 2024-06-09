import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, mean_absolute_error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import os
import gdown
import zipfile

# Function to download data from Google Drive
def download_data():
    url = 'https://drive.google.com/uc?id=15p0yrMlhRMBKlui_zwwzQkez4V6Wp6a6'
    output = 'linkedin-job-postings.zip'
    gdown.download(url, output, quiet=False)
    
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall('.')

# Function to load data
@st.cache_data
def load_data():
    # Check if data is already downloaded, if not, download it
    if not os.path.exists('postings.csv'):
        download_data()
    
    # Load all datasets
    postings = pd.read_csv('postings.csv')
    industries = pd.read_csv('mappings/industries.csv')
    skills = pd.read_csv('mappings/skills.csv')
    benefits = pd.read_csv('jobs/benefits.csv')
    job_industries = pd.read_csv('jobs/job_industries.csv')
    job_skills = pd.read_csv('jobs/job_skills.csv')
    salaries = pd.read_csv('jobs/salaries.csv')
    companies = pd.read_csv('companies/companies.csv')
    employee_counts = pd.read_csv('companies/employee_counts.csv')
    company_industries = pd.read_csv('companies/company_industries.csv')
    company_specialities = pd.read_csv('companies/company_specialities.csv')
    
    return postings, industries, skills, benefits, job_industries, job_skills, salaries, companies, employee_counts, company_industries, company_specialities

# Ensure data is downloaded
if not os.path.exists('postings.csv'):
    download_data()

# Load data
postings, industries, skills, benefits, job_industries, job_skills, salaries, companies, employee_counts, company_industries, company_specialities = load_data()

# Data Merging
merged_jobs = pd.merge(postings, benefits, on='job_id', how='left')
merged_jobs = pd.merge(merged_jobs, job_industries, on='job_id', how='left')
merged_jobs = pd.merge(merged_jobs, job_skills, on='job_id', how='left')
merged_jobs = pd.merge(merged_jobs, salaries, on='job_id', how='left')

merged_jobs = pd.merge(merged_jobs, industries, on='industry_id', how='left')
merged_jobs = pd.merge(merged_jobs, skills, on='skill_abr', how='left')

merged_companies = pd.merge(companies, employee_counts, on='company_id', how='left')
merged_companies = pd.merge(merged_companies, company_industries, on='company_id', how='left')
merged_companies = pd.merge(merged_companies, company_specialities, on='company_id', how='left')

comprehensive_data = pd.merge(merged_jobs, merged_companies, on='company_id', how='left')
comprehensive_data = comprehensive_data.drop_duplicates(subset='job_id', keep='first')

# Handling Missing Values
cols_fill_zero = ['applies', 'views', 'follower_count', 'employee_count', "remote_allowed"]
for col in cols_fill_zero:
    comprehensive_data[col].fillna(0, inplace=True)
comprehensive_data['closed_time'].fillna("Still Open", inplace=True)
comprehensive_data['inferred'].fillna("Unknown", inplace=True)
comprehensive_data['description_y'].fillna("Not Specified", inplace=True)
object_columns = comprehensive_data.select_dtypes(include=['object']).columns
comprehensive_data[object_columns] = comprehensive_data[object_columns].fillna('Not Given')

# Streamlit App
st.title("LinkedIn Job Posting Analysis")

# Display Data
st.subheader("Raw Data")
st.write(comprehensive_data.head())

# Data Visualization
st.subheader("Data Visualization")

# Distribution of Jobs by Work Type
work_type_distribution = comprehensive_data['formatted_work_type'].value_counts()
fig, ax = plt.subplots()
work_type_distribution.plot(kind='bar', color='skyblue', ax=ax)
ax.set_title('Distribution of Jobs by Work Type')
ax.set_xlabel('Work Type')
ax.set_ylabel('Number of Jobs')
plt.xticks(rotation=45)
st.pyplot(fig)

# Top 10 Most Common Job Titles by Median Salary
top_titles = comprehensive_data['title'].value_counts().index[:10]
top_titles_data = comprehensive_data[comprehensive_data['title'].isin(top_titles)]
median_salaries = top_titles_data.groupby('title')['med_salary_x'].mean().sort_values(ascending=False)
fig, ax = plt.subplots()
median_salaries.plot(kind='bar', color='lightcoral', ax=ax)
ax.set_title('Median Salary for Top 10 Most Common Job Titles')
ax.set_xlabel('Job Title')
ax.set_ylabel('Mean Salary')
plt.xticks(rotation=45)
st.pyplot(fig)

# Top 10 Companies with Most Job Postings
top_companies = comprehensive_data['name'].value_counts().head(10)
fig, ax = plt.subplots()
top_companies.plot(kind='bar', color='lightsalmon', ax=ax)
ax.set_title('Top 10 Companies with Most Job Postings')
ax.set_xlabel('Company Name')
ax.set_ylabel('Number of Job Postings')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Distribution of Remote Work Options
remote_work_distribution = comprehensive_data['remote_allowed'].value_counts()
fig, ax = plt.subplots()
remote_work_distribution.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['lightcoral', 'lightgreen', 'lightblue'], ax=ax)
ax.set_title('Distribution of Remote Work Options')
plt.ylabel('')
st.pyplot(fig)

# Top 10 Cities with Most Job Postings
top_cities = comprehensive_data['city'].value_counts().head(10)
top_states = comprehensive_data['state'].value_counts().head(10)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
top_cities.plot(kind='bar', color='lightblue', ax=ax1)
ax1.set_title('Top 10 Cities with Most Job Postings')
ax1.set_xlabel('City')
ax1.set_ylabel('Number of Job Postings')
ax1.set_xticks(ax1.get_xticks())
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
top_states.plot(kind='bar', color='lightgreen', ax=ax2)
ax2.set_title('Top 10 States with Most Job Postings')
ax2.set_xlabel('State')
ax2.set_ylabel('Number of Job Postings')
ax2.set_xticks(ax2.get_xticks())
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Most In-Demand Skills
top_skills = comprehensive_data['skill_abr'].value_counts().head(10)
fig, ax = plt.subplots()
top_skills.plot(kind='bar', color='lightcoral', ax=ax)
ax.set_title('Top 10 Most In-Demand Skills')
ax.set_xlabel('Skill Abbreviation')
ax.set_ylabel('Number of Mentions in Job Postings')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Distribution of Job Postings by Experience Level
experience_level_distribution = comprehensive_data['formatted_experience_level'].value_counts()
median_salary_by_experience = comprehensive_data.groupby('formatted_experience_level')['med_salary_x'].median()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
experience_level_distribution.plot(kind='bar', color='lightgreen', ax=ax1)
ax1.set_title('Distribution of Job Postings by Experience Level')
ax1.set_xlabel('Experience Level')
ax1.set_ylabel('Number of Job Postings')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
median_salary_by_experience.sort_values().plot(kind='bar', color='lightcoral', ax=ax2)
ax2.set_title('Median Salary by Experience Level')
ax2.set_xlabel('Experience Level')
ax2.set_ylabel('Median Salary')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Further Experiments Independent From Article
st.subheader("Further Experiments Independent From Article")

# Text Classification Pipeline
st.write("Pipeline for Text Classification")

# Logistic Regression for 'Hospitals and Health Care' Industry
st.write("Logistic Regression for 'Hospitals and Health Care' Industry")
df = comprehensive_data.copy()
df['is_healthcare'] = (df['industry_name'] == 'Hospitals and Health Care').astype(int)
df['text_data'] = df['title'] + ' ' + df['description_x']
df['text_data'].fillna('', inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df['text_data'], df['is_healthcare'], test_size=0.2, random_state=42)
pipeline = make_pipeline(TfidfVectorizer(stop_words='english'), LogisticRegression(random_state=42))
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

st.write("Accuracy:", accuracy_score(y_test, y_pred))

# Display Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.table(report_df)

# Feature Weights Extraction
st.write("Feature Weights Extraction")
vectorizer = pipeline.named_steps['tfidfvectorizer']
idf_scores = vectorizer.idf_
feature_names = vectorizer.get_feature_names_out()
idf_df = pd.DataFrame({'term': feature_names, 'idf_score': idf_scores})
idf_df_sorted = idf_df.sort_values(by='idf_score', ascending=False)
st.write("Top 10 terms with highest IDF scores:", idf_df_sorted.head(10))

# Logistic Regression for 'Retail' Industry
st.write("Logistic Regression for 'Retail' Industry")
df['is_retail'] = (df['industry_name'] == 'Retail').astype(int)
X_train, X_test, y_train, y_test = train_test_split(df['text_data'], df['is_retail'], test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

st.write("Accuracy:", accuracy_score(y_test, y_pred))

# Display Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.table(report_df)

# Multi-Class Classification for Top 10 Industries
st.write("Multi-Class Classification for Top 10 Industries")
top_10_industries = df['industry_name'].value_counts().nlargest(10).index
df_top_10 = df[df['industry_name'].isin(top_10_industries)]
df_top_10['text_data'] = df_top_10['title'] + ' ' + df_top_10['description_x']
df_top_10['text_data'].fillna('', inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df_top_10['text_data'], df_top_10['industry_name'], test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

st.write("Classification Report:")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.table(report_df)

# Extract and show feature weights
vectorizer = pipeline.named_steps['tfidfvectorizer']
classifier = pipeline.named_steps['logisticregression']
feature_names = vectorizer.get_feature_names_out()
coefficients = classifier.coef_
for class_index, class_name in enumerate(classifier.classes_):
    top10 = np.argsort(coefficients[class_index])[-10:][::-1]
    st.write(f"Class: {class_name}")
    st.write("Top 10 features:")
    feature_data = {
        "Feature": [feature_names[i] for i in top10],
        "Coefficient": [coefficients[class_index][i] for i in top10]
    }
    feature_df = pd.DataFrame(feature_data)
    st.table(feature_df)
    st.write("\n")

# Salary Prediction
st.subheader("Salary Prediction")

# Preprocess salary values
def convert_hourly_to_yearly(row):
    if row['pay_period_x'] == 'HOURLY':
        row['max_salary_x'] = row['max_salary_x'] * 2080
        row['min_salary_x'] = row['min_salary_x'] * 2080
        if not np.isnan(row['med_salary_x']):
            row['med_salary_x'] = row['med_salary_x'] * 2080
        row['pay_period_x'] = "YEARLY"
    return row

# Apply conversion to max_salary_x, min_salary_x, and med_salary_x
comprehensive_data = comprehensive_data.apply(convert_hourly_to_yearly, axis=1)

# Calculate med_salary_x where it is NaN
comprehensive_data['med_salary_x'] = comprehensive_data.apply(
    lambda row: (row['max_salary_x'] + row['min_salary_x']) / 2 if np.isnan(row['med_salary_x']) else row['med_salary_x'],
    axis=1
)

# Remove extreme outliers from salary columns
Q1 = comprehensive_data['med_salary_x'].quantile(0.25)
Q3 = comprehensive_data['med_salary_x'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

comprehensive_data = comprehensive_data[
    (comprehensive_data['med_salary_x'] >= lower_bound) & (comprehensive_data['med_salary_x'] <= upper_bound)
]

# Additional feature engineering
comprehensive_data['title_length'] = comprehensive_data['title'].apply(lambda x: len(str(x)))
comprehensive_data['description_length'] = comprehensive_data['description_x'].apply(lambda x: len(str(x)))

# Feature Selection: Select relevant features for predicting the median salary
features = ['title', 'description_x', 'location', 'company_name', 'formatted_work_type', 'skills_desc', 'formatted_experience_level', 'views', 'applies', 'employee_count', 'follower_count', 'title_length', 'description_length']
target = 'med_salary_x'

# Handling missing values and encoding categorical features
# Separate features into numerical and categorical
num_features = ['views', 'applies', 'employee_count', 'follower_count', 'title_length', 'description_length']
cat_features = ['title', 'location', 'company_name', 'formatted_work_type', 'skills_desc', 'formatted_experience_level']

# Preprocessing for numerical data: impute missing values and scale
num_transformer = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)

# Preprocessing for categorical data: impute missing values and encode
cat_transformer = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='missing'),
    OneHotEncoder(handle_unknown='ignore')
)

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)

# Define the model
model = GradientBoostingRegressor(n_estimators=200, random_state=42)

# Create and evaluate the pipeline
pipeline = make_pipeline(preprocessor, model)

# Split the data into training and testing sets
X = comprehensive_data[features]
y = comprehensive_data[target]

# Drop rows where target is missing
X = X[y.notna()]
y = y.dropna()

# Use cross-validation to evaluate the model
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
st.write(f'Cross-Validated MAE: {-cv_scores.mean()}')

# Fit the model
pipeline.fit(X, y)

# Predict on the test set
y_pred = pipeline.predict(X)

# Evaluate the model
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

st.write(f"Mean Absolute Error (MAE): {mae}")
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"Root Mean Squared Error (RMSE): {rmse}")
st.write(f"R-squared (R2): {r2}")

# Plotting the actual vs predicted values
fig, ax = plt.subplots()
ax.scatter(y, y_pred, alpha=0.3)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Actual vs Predicted Median Salaries')
st.pyplot(fig)
