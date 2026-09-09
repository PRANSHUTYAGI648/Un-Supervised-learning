import pandas as pd

# Load the dataset
df = pd.read_csv("dataset/Mall_Customers.csv")

# Display the first 5 rows
print("First 5 Rows:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
df.info()

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display the number of rows and columns
print("\nDataset Shape:")
print(df.shape)
# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check for duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Display basic statistical information
print("\nStatistical Summary:")
print(df.describe())
# Select features for customer segmentation
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Display the selected features
print("\nSelected Features:")
print(X.head())

# Display the shape of selected features
print("\nSelected Features Shape:")
print(X.shape)
# Import KMeans
from sklearn.cluster import KMeans

# Calculate WCSS for different numbers of clusters
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

print("\nWCSS Values:")
print(wcss)
import matplotlib.pyplot as plt

# Plot the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()
# Create the K-Means model
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

# Train the model
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_

# Add cluster labels to the dataset
df["Cluster"] = labels

# Display the first 10 customers with their cluster
print("\nCustomer Clusters:")
print(df.head(10))

# Display the number of customers in each cluster
print("\nCustomers in Each Cluster:")
print(df["Cluster"].value_counts().sort_index())
# Get cluster centers
centers = kmeans.cluster_centers_

# Plot the customer clusters
plt.figure(figsize=(10, 6))

plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=labels,
    s=80
)

# Plot cluster centers
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    s=200,
    marker="X",
    label="Cluster Centers"
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)

plt.show()
# Import silhouette score
from sklearn.metrics import silhouette_score

# Calculate the Silhouette Score
silhouette_avg = silhouette_score(X, labels)

# Display the Silhouette Score
print("\nSilhouette Score:")
print(silhouette_avg)
# Analyze each customer cluster
cluster_analysis = df.groupby("Cluster")[[
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]].mean()

print("\nCluster Analysis:")
print(cluster_analysis)
import joblib

# Save the trained K-Means model
joblib.dump(kmeans, "model/model.pkl")

# Save the feature names
joblib.dump(
    ["Annual Income (k$)", "Spending Score (1-100)"],
    "model/features.pkl"
)

print("\nModel saved successfully!")
print("Model file: model/model.pkl")
# ==============================
# K-Means Clustering
# ==============================

# Set the optimal number of clusters
optimal_k = 5

# Create the K-Means model
kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

# Train the model
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_

# Add cluster labels to the dataset
df["Cluster"] = labels

# Display cluster counts
print("\nCustomer Cluster Counts:")
print(df["Cluster"].value_counts().sort_index())

# Display cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)