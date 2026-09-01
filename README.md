# Data Science Course Progress and Practical Portfolio

Created by DEV_MANISH

GitHub: https://github.com/DEVELOPER-MANISH007
This repository is a structured learning portfolio for a Data Science and Machine Learning course. It contains Python notebooks, scripts, datasets, and small web applications that cover the full journey from basic Python and data handling to exploratory data analysis, feature engineering, and machine learning deployment.

---

## Overview

The workspace is organized by topic and follows a practical progression:

1. Python foundations and data manipulation
2. Data visualization and database basics
3. Web app development with Flask and Streamlit
4. Exploratory Data Analysis (EDA)
5. Feature engineering and preprocessing
6. Classical machine learning algorithms
7. End-to-end ML project implementation

---

## Course Structure

### 1. Basics
Folder: [01.BASICS](01.BASICS)

This section includes introductory work with:
- Python notebooks for pandas, seaborn, and SQLite
- Basic data handling and visualization
- A sample student dataset from [01.BASICS/students_dataset.csv](01.BASICS/students_dataset.csv)

### 2. Multi-threading in Python
Folder: [02.MULTI THREADING IN PYTHON](02.MULTI%20THREADING%20IN%20PYTHON)

Contains practical examples of:
- Threading
- Parallel execution concepts
- Basic performance comparison with concurrent tasks

### 3. Memory Management
Folder: [03.MEMORY_MGMT](03.MEMORY_MGMT)

Focuses on memory-related concepts and notebook-based practice.

### 4. Flask
Folder: [04.FLASK](04.FLASK)

Includes a simple Flask application in [04.FLASK/main.py](04.FLASK/main.py) that demonstrates:
- Creating a basic web server
- Defining routes
- Running a local app with Flask

### 5. Streamlit
Folder: [05.stremlit](05.stremlit)

Includes interactive dashboard examples using Streamlit:
- [05.stremlit/app.py](05.stremlit/app.py) for a simple dashboard
- [05.stremlit/classification.py](05.stremlit/classification.py) for an Iris flower classification example using scikit-learn

### 6. Statistics and Inferential Statistics
Folders: [06.STATISTICS](06.STATISTICS) and [07.Inferential Statistics](07.Inferential%20Statistics)

These folders are intended for statistical theory and practical understanding of:
- Descriptive statistics
- Probability concepts
- Inferential testing and analysis

### 7. Feature Engineering
Folder: [08.Feature Engineerings](08.Feature%20Engineerings)

This section contains notebooks on:
- Handling missing values
- Dealing with imbalanced datasets
- SMOTE for resampling
- Outlier handling
- One-hot encoding and ordinal encoding

### 8. Exploratory Data Analysis (EDA)
Folder: [09.EDA](09.EDA)

This is one of the most practical sections of the course. It includes EDA work on real datasets such as:
- [09.EDA/winequality-red.csv](09.EDA/winequality-red.csv)
- [09.EDA/googleplaystore.csv](09.EDA/googleplaystore.csv)
- [09.EDA/google_cleaned_data.csv](09.EDA/google_cleaned_data.csv)
- [09.EDA/flight_price.xlsx](09.EDA/flight_price.xlsx)

These notebooks cover:
- Data cleaning
- Visualization
- Pattern discovery
- Feature selection and preparation for modeling

### 9. Machine Learning Foundations
Folder: [10.Into to macchine learning](10.Into%20to%20macchine%20learning)

This section introduces the core concepts of machine learning and the transition from data analysis to modeling.

### 10. Regression and ML Lifecycle Projects
Folder: [11.Understand complete linear Regression Indepth Intution And Praccticals](11.Understand%20complete%20linear%20Regression%20Indepth%20Intution%20And%20Praccticals)

Includes practical regression notebooks and related materials.

### 11. Regularization Techniques
Folder: [12.Section 28 Ridge, Lasso And ElasticNet ML](12.Section%2028%20Ridge,%20Lasso%20And%20ElasticNet%20ML)

Focuses on:
- Ridge regression
- Lasso regression
- ElasticNet

### 12. End-to-End ML Project Implementation
Folder: [13. step by step Project Implementation wth lifecycle  of ML](13.%20step%20by%20step%20Project%20Implementation%20wth%20lifecycle%20%20of%20ML)

This section contains a full machine learning lifecycle project and is one of the strongest examples in the repository. A detailed project README is available here:
- [13. step by step Project Implementation wth lifecycle  of ML/004 APP.PY/README.md](13.%20step%20by%20step%20Project%20Implementation%20wth%20lifecycle%20%20of%20ML/004%20APP.PY/README.md)

The project demonstrates:
- Data preparation
- Model training
- Feature scaling
- Regression modeling
- Flask-based deployment

### 13. Classification Algorithms
Folders:
- [14.Logistic Regression  (Binary Classification)](14.Logistic%20Regression%20%20(Binary%20Classification))
- [15.Support Vector Machine](15.Support%20Vector%20Machine)
- [16. Naive Bays Theorm](16.%20Naive%20Bays%20Theorm)
- [17. K- Nearest neighbour Ml Algorithm](17.%20K-%20Nearest%20neighbour%20Ml%20Algorithm)
- [18. Decision Tree Classifier And Regresssor](18.%20Decision%20Tree%20Classifier%20And%20Regresssor)
- [19.Randm Forest Machine Learning](19.Randm%20Forest%20Machine%20Learning)
- [20.Adaboost Machine Learning Algorithm](20.Adaboost%20Machine%20Learning%20Algorithm)
- [21.Grediant Boosting](21.Grediant%20Boosting)
- [22.XGboost](22.XGboost)
- [23.Unsupervised Learning](23.Unsupervised%20Learning)

These folders cover the common supervised and unsupervised learning methods used in real-world data science projects.

---

## Key Datasets Used

This repository includes several practical datasets:

- [01.BASICS/students_dataset.csv](01.BASICS/students_dataset.csv)
  - Student performance-style data with academic scores and demographic features
- [09.EDA/winequality-red.csv](09.EDA/winequality-red.csv)
  - Wine quality data for regression/classification experimentation
- [09.EDA/googleplaystore.csv](09.EDA/googleplaystore.csv)
  - Google Play Store app metadata for EDA and feature analysis
- [09.EDA/google_cleaned_data.csv](09.EDA/google_cleaned_data.csv)
  - Cleaned version of the Google Play Store data
- [09.EDA/flight_price.xlsx](09.EDA/flight_price.xlsx)
  - Flight price dataset for exploratory analysis and feature engineering

---

## Practical Projects Included

### Flask App
Run the basic Flask server from [04.FLASK/main.py](04.FLASK/main.py):

```bash
cd "04.FLASK"
python main.py
```

### Streamlit Dashboards
Run the Streamlit examples from [05.stremlit](05.stremlit):

```bash
cd "05.stremlit"
streamlit run app.py
```

Or:

```bash
cd "05.stremlit"
streamlit run classification.py
```

### Jupyter Notebooks
Most of the learning material is stored in notebooks inside the course folders. These can be opened in Jupyter Notebook or JupyterLab.

---

## Tools and Libraries Used

The notebooks and scripts mainly use:
- Python
- pandas
- numpy
- seaborn
- scikit-learn
- Flask
- Streamlit
- Jupyter Notebook
- SQLite

---

## Learning Journey Summary

This course portfolio demonstrates steady progression from:
- basic Python and data handling,
- to visualization and EDA,
- to feature engineering,
- and finally into machine learning algorithms and deployment.

It is a strong example of how a beginner-friendly course can evolve into practical, real-world machine learning work.

---

## Notes

This repository is best viewed as a study archive and practical portfolio rather than a single production-ready project. Each folder represents a milestone in the learning path, and the strongest end-to-end example is the ML lifecycle project in [13. step by step Project Implementation wth lifecycle  of ML](13.%20step%20by%20step%20Project%20Implementation%20wth%20lifecycle%20%20of%20ML).

---

## Suggested Next Steps

To continue the learning journey:
- Review the notebooks in order from basics to advanced ML topics
- Re-run the Streamlit and Flask examples
- Practice EDA on the provided datasets
- Build your own mini-project using one of the machine learning algorithms covered here
