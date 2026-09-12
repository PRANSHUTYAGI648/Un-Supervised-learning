# Customer Segmentation using K-Means Clustering

## Project Overview

Customer Segmentation is a Machine Learning project that groups customers into different segments based on their income and spending behavior.

This project uses the K-Means Clustering algorithm to identify similar groups of customers.

A Flask web dashboard is also developed to visualize the clusters and predict the cluster of a new customer.

## Objective

The main objectives of this project are:

- To analyze customer income and spending behavior.
- To divide customers into meaningful groups.
- To use K-Means Clustering for customer segmentation.
- To visualize different customer clusters.
- To predict the cluster of a new customer through a web dashboard.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Flask
- Joblib
- HTML
- CSS

## Dataset

The project uses the Mall Customers dataset.

The dataset contains information such as:

- Customer ID
- Gender
- Age
- Annual Income
- Spending Score

For clustering, the following two features are used:

- Annual Income (k$)
- Spending Score (1-100)

## Machine Learning Algorithm

### K-Means Clustering

K-Means is an unsupervised Machine Learning algorithm used to divide data into groups called clusters.

In this project, K-Means is used to group customers based on:

- Annual Income
- Spending Score

## Elbow Method

The Elbow Method was used to determine the suitable number of clusters.

The analysis showed that:

**Optimal Number of Clusters = 5**

## Model Evaluation

The clustering model was evaluated using the Silhouette Score.

**Silhouette Score = 0.55**

A score around 0.55 indicates reasonably good separation between the customer clusters.

## Customer Segments

The customers are divided into five clusters:

### Cluster 0
Medium-income customers with moderate spending behavior.

### Cluster 1
High-income customers with high spending behavior and strong customer value.

### Cluster 2
Low-income customers with high spending behavior and strong buying interest.

### Cluster 3
High-income customers with low spending behavior who may need targeted offers.

### Cluster 4
Low-income customers with low spending behavior and limited purchasing activity.

## Project Features

- Customer data analysis
- K-Means clustering
- Elbow Method visualization
- Customer cluster visualization
- Silhouette Score evaluation
- Trained model saving
- Flask web dashboard
- New customer cluster prediction
- Customer segment descriptions

## Project Structure

```text
un-supervised learning
│
├── dataset
│   └── Mall_Customers.csv
│
├── model
│   ├── model.pkl
│   └── features.pkl
│
├── static
│   ├── elbow_method.png
│   └── customer_clusters.png
│
├── templates
│   └── index.html
│
├── app.py
├── model.py
├── README.md
└── requirements.txt