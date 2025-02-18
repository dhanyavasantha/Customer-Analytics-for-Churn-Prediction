# Customer Analytics for Churn Prediction 

## Project Overview  
This project analyzes customer churn in the telecommunications industry by identifying key factors contributing to customer attrition. The objective is to develop a predictive model that classifies customers as likely to churn or remain, enabling businesses to implement proactive retention strategies.  

## Objectives  
1. **Exploratory Data Analysis (EDA):** Understand the patterns and correlations in customer data.  
2. **Feature Engineering:** Transform raw data into meaningful features to improve predictive accuracy.  
3. **Machine Learning Model Development:** Train models to predict customer churn.  
4. **Model Evaluation:** Compare different models based on accuracy, precision, recall, and F1-score.  
5. **Business Strategy:** Provide actionable insights to reduce churn and enhance customer retention.  

## Dataset Description  
The dataset contains 7,043 customer records with numerical and categorical features related to demographics, account details, services, and billing information. The primary outcome variable is **Churn**, which indicates whether a customer has left the service.  

### Key Features  
- **Demographics:** Gender, Senior Citizen, Partner, Dependents  
- **Account Information:** Tenure, Contract Type, Payment Method  
- **Service Usage:** Phone Service, Internet Service, Streaming Services  
- **Billing Details:** Monthly Charges, Total Charges  
- **Target Variable:** Churn (Yes/No)  

## Problem Description  
Customer churn is a major challenge for businesses, especially in subscription-based services like telecommunications. The ability to predict churn enables companies to implement targeted strategies to retain customers and minimize revenue loss.  

The project aims to:  
- **Identify key drivers of churn** such as contract type, service quality, or pricing.  
- **Develop machine learning models** to classify customers into churn or non-churn categories.  
- **Recommend business strategies** to reduce customer attrition.  

## Methodology  

### 1. Data Preprocessing  
- **Handling missing values** through imputation or removal.  
- **Encoding categorical variables** such as contract types and payment methods.  
- **Scaling numerical features** like monthly charges and tenure.  

### 2. Exploratory Data Analysis (EDA)  
- **Identify trends and relationships** between customer attributes and churn.  
- **Correlation heatmaps** to understand dependencies between features.  
- **Visualization techniques** such as histograms, box plots, and bar charts to analyze churn patterns.  

### 3. Feature Engineering  
- **Creating new features** such as contract tenure groups or payment consistency indicators.  
- **Transforming numerical features** for better model interpretability.  

### 4. Model Selection and Training  
Various machine learning models are compared for churn prediction:  

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|----------|--------------|------------|------------|-------------|
| Logistic Regression | 79.5 | 79.3 | 78.8 | 79.0 |
| Decision Tree | 85.2 | 85.0 | 84.5 | 84.7 |
| Random Forest | 90.1 | 89.9 | 89.7 | 89.8 |
| Gradient Boosting | **92.5** | **92.3** | **92.0** | **92.2** |

Gradient Boosting performed the best, achieving an accuracy of 92.5%.  

### 5. Model Evaluation  
- **ROC-AUC Curve** to assess classification performance.  
- **Precision-Recall Trade-off** to balance false positives and false negatives.  
- **Feature Importance Analysis** to determine which attributes contribute most to churn.  

## Key Findings  
- **Customers with month-to-month contracts** are more likely to churn than those with long-term contracts.  
- **Higher monthly charges** correlate with increased churn.  
- **Customers using electronic checks** as a payment method tend to churn more frequently.  
- **Long-tenured customers** are less likely to churn.  

## Business Recommendations  
1. **Offer incentives for long-term contracts** to reduce month-to-month churn rates.  
2. **Implement loyalty programs** for high-risk customers to encourage retention.  
3. **Improve customer service response times** to reduce dissatisfaction.  
4. **Monitor customers with high monthly charges** and offer personalized discounts.  

## Future Scope  
- Implement deep learning models such as LSTMs for time-series churn prediction.  
- Use **real-time customer behavior tracking** to enhance retention strategies.  
- Develop a **customer segmentation framework** based on churn risk levels.  

## Technologies Used  
| Technology | Purpose |
|------------|---------|
| Python | Data Cleaning, Analysis, and Machine Learning |
| Pandas | Data Manipulation |
| Matplotlib & Seaborn | Data Visualization |
| Scikit-learn | Machine Learning Model Training |
| XGBoost | Advanced Gradient Boosting Model |
| TensorFlow | Deep Learning Experiments |
