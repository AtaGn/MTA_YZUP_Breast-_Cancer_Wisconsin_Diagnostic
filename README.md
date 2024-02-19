# Breast Cancer Diagnosis with Machine Learning

This project is an interactive web application built with Streamlit, designed to diagnose breast cancer using various machine learning models. It utilizes the Python libraries pandas for data manipulation, seaborn and matplotlib for data visualization, and scikit-learn for machine learning. The application guides the user through several steps: data loading, preprocessing, model selection, training, and analysis.

## Features

- **Interactive Web UI**: Built using Streamlit, allowing for easy navigation and operation.
- **Data Preprocessing**: Cleans the dataset by removing unnecessary columns and encoding categorical data.
- **Visualization**: Generates correlation heatmaps and scatter plots to visualize the data.
- **Model Selection**: Offers a choice between KNN, SVM, and Naive Bayes classifiers.
- **Model Training**: Includes parameter tuning with GridSearchCV for KNN and SVM models.
- **Model Analysis**: Displays the model's accuracy, precision, recall, F1-score, and a confusion matrix to evaluate performance.

## Usage

1. **Load Data**: Use the sidebar to upload a breast cancer dataset in CSV format.
2. **Preprocess Data**: The application automatically preprocesses the data upon loading.
3. **Select and Train Model**: Choose a model and adjust its parameters. The app trains the model with the dataset.
4. **Analyze Model**: Review the model's performance metrics and confusion matrix.

**github link**: https://github.com/AtaGn/MTA_YZUP_Breast-_Cancer_Wisconsin_Diagnostic

**online link**: https://mta-yzup-breast-cancer-wisconsin-diagnostic.streamlit.app/
