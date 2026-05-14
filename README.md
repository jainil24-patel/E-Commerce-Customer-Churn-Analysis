Machine Learning-Based Customer Churn Prediction and Customer Segmentation Using RFM Analytics in E-Commerce
Project Overview

Customer retention is one of the biggest challenges in the e-commerce industry. Businesses often lose customers due to changing customer behavior, reduced engagement, dissatisfaction, and competition. Identifying customers who are likely to churn and understanding customer purchasing patterns can help organizations improve retention strategies and increase profitability.

This project presents a machine learning-based framework for:

Customer Churn Prediction
Customer Segmentation
RFM Analytics Implementation
Business Insight Generation

The project combines supervised and unsupervised machine learning techniques to analyze customer behavior and support data-driven business decision-making.

Project Objectives

The main objectives of this project are:

Predict customer churn using machine learning
Segment customers into meaningful groups
Apply RFM analytics for customer behavior analysis
Generate business insights from customer data
Support customer retention strategies
Dataset Information

Dataset Source:

https://www.kaggle.com/datasets/samuelsemaya/e-commerce-customer-churn

The dataset contains customer-related information including:

Tenure
CashbackAmount
WarehouseToHome
DaySinceLastOrder
NumberOfAddress
SatisfactionScore
PreferredOrderCat
MaritalStatus
Complain
Customer Churn Status

Target Variable:

Churn

1 = Customer Churned
0 = Customer Retained
Technologies Used

Programming Language:

Python

Libraries:

Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Joblib

Development Environment:

Jupyter Notebook
Project Workflow
Step 1: Data Collection
Load E-Commerce customer churn dataset
Step 2: Data Preprocessing
Handle missing values
Encode categorical variables
Clean and prepare data
Step 3: Feature Engineering Using RFM Analytics

RFM:

Recency → DaySinceLastOrder
Frequency → Tenure
Monetary → CashbackAmount
Step 4: Data Scaling
StandardScaler implementation
Step 5: Customer Segmentation

Algorithm Used:

K-Means Clustering

Method:

Elbow Method for selecting cluster count

Selected Number of Clusters:

4

Step 6: Customer Churn Prediction

Algorithm Used:

Random Forest Classifier

Train-Test Split:

80:20

Step 7: Model Evaluation

Evaluation Metrics:

Accuracy
Precision
Recall
F1-Score
ROC-AUC
Exploratory Data Analysis

The following analyses were performed:

Statistical Summary Analysis
Missing Value Analysis
Customer Churn Distribution
Customer Engagement Analysis
RFM Behavior Analysis
Correlation Heatmap Analysis
Algorithms Used
K-Means Clustering

Purpose:

Segment customers into groups based on purchasing behavior

Advantages:

Simple and efficient
Suitable for customer segmentation
Supports behavioral analysis
Random Forest Classifier

Purpose:

Predict customer churn behavior

Advantages:

High predictive accuracy
Handles structured datasets effectively
Reduces overfitting
Generates feature importance analysis
Results

Customer Churn Prediction Accuracy:

94%

ROC-AUC Score:

0.98

Number of Customer Segments:

4

Major Important Features:

Tenure
Frequency
CashbackAmount
Monetary
WarehouseToHome
Complaint Status
Generated Visualizations
Elbow Method Graph
Feature Importance Graph
ROC Curve
Correlation Heatmap
Customer Segmentation Outputs
Business Applications

This project can help businesses:

Identify customers likely to churn
Improve customer retention
Create personalized marketing strategies
Improve customer engagement
Support customer relationship management
Increase profitability
Support data-driven decision-making
Project Structure

E-Commerce-Customer-Churn-Analysis │ ├── Dataset/ ├── Source_Code/ ├── Models/ ├── Images/ ├── Project_Report/ ├── README.md

Future Improvements

Future enhancements may include:

Deep learning implementation
Real-time prediction dashboard
Cloud deployment
Web application integration
Advanced recommendation systems
GitHub Repository

Repository Link:

https://github.com/jainil24-patel/E-Commerce-Customer-Churn-Analysis

Author

Mr. Patel

MSc IT for Business Data Analytics

Machine Learning-Based Customer Churn Prediction and Customer Segmentation Using RFM Analytics in E-Commerce