# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load cleaned datasets (from FA-1)
cab_rides = pd.read_csv("cleaned_cab_rides.csv")
weather = pd.read_csv("cleaned_weather.csv")

# ================================
# 1️⃣ Exploratory Data Analysis (EDA)
# ================================

# Histogram for price distribution
plt.figure(figsize=(8, 5))
sns.histplot(cab_rides["price"], bins=30, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# Boxplot comparing Uber vs. Lyft fares
plt.figure(figsize=(8, 5))
sns.boxplot(x=cab_rides["cab_type"], y=cab_rides["price"])
plt.title("Uber vs. Lyft Fare Comparison")
plt.xlabel("Cab Type")
plt.ylabel("Price")
plt.show()

# ================================
# 2️⃣ Association Rule Mining
# ================================

# Convert categorical attributes to one-hot encoding
encoded_data = pd.get_dummies(cab_rides[["cab_type", "source", "destination", "peak_time"]])

# Apply Apriori Algorithm
frequent_itemsets = apriori(encoded_data, min_support=0.1, use_colnames=True)

# Generate Association Rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Display top rules
print("\nTop Association Rules:\n", rules.head())

# Save rules to CSV for further analysis
rules.to_csv("association_rules.csv", index=False)

# ================================
# 3️⃣ K-Means Clustering
# ================================

# Select numerical features for clustering
features = cab_rides[["price", "distance", "surge_multiplier"]]

# Check for missing values
print("\nMissing values before handling:")
print(features.isnull().sum())

# Fill missing values with the median
features = features.fillna(features.median())

# Standardize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Verify missing values after scaling
print("\nMissing values after filling:")
print(pd.DataFrame(features_scaled, columns=features.columns).isnull().sum())

# Determine the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker="o")
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# Apply K-Means with optimal clusters (e.g., k=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cab_rides["cluster"] = kmeans.fit_predict(features_scaled)

# Save the clustered dataset
cab_rides.to_csv("clustered_cab_rides.csv", index=False)
print("\nClustered Data Sample:\n", cab_rides.head())

print("\n✅ FA-2 Completed! Files Generated:")
print("📁 association_rules.csv (Association Mining Output)")
print("📁 clustered_cab_rides.csv (K-Means Clustering Output)")

# Load the dataset
cab_rides = pd.read_csv("cleaned_cab_rides.csv")  # Ensure dataset is preprocessed

# Reset index in original DataFrame to keep track of rows
cab_rides = cab_rides.reset_index()

# Select relevant features for clustering
features = cab_rides[['distance', 'price']].copy()

# Drop rows with NaN values
filtered_data = features.dropna().reset_index()  # Reset index after dropping NaNs

# Standardizing the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(filtered_data[['distance', 'price']])

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # `n_init=10` to avoid warnings
filtered_data['Cluster'] = kmeans.fit_predict(scaled_features)

# Merge back into original dataset using 'index'
cab_rides = cab_rides.merge(filtered_data[['index', 'Cluster']], on='index', how='left')

# Scatter plot of clusters
plt.figure(figsize=(8, 6))
plt.scatter(filtered_data['distance'], filtered_data['price'], c=filtered_data['Cluster'], cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0] * scaler.scale_[0] + scaler.mean_[0], 
            kmeans.cluster_centers_[:, 1] * scaler.scale_[1] + scaler.mean_[1], 
            color='red', marker='X', s=200, label='Centroids')

# Labels & Titles
plt.xlabel("Distance (miles)")
plt.ylabel("Price (USD)")
plt.title("K-Means Clustering of Cab Rides")
plt.legend()
plt.show()
