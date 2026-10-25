# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("Online retail_new.csv", header=None)
df.sample(5)

# %%
df.head(10)

# %%
df.tail(10)

# %%
df.shape

# %%
for column in df.columns:
    df[column] = df[column].str.strip()

# %%
df.sample(10)

# %%
df = df.replace("", pd.NA)

# %%
df.sample(5)

# %%
# Check missing values
print("Missing values:\n",df.isnull().sum().sum())

# Check duplicate transactions
print("Duplicate rows:", df.duplicated().sum())

# %%


# Remove completely empty rows
df = df.dropna(how="all")

# Check missing values and duplicates
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

# Display cleaned data
display(df.head())


# %%

# Transaction statistics
print("Total transactions:", len(df))

# Products in each transaction
print("\nTransaction size statistics:")
k=df.count(axis=1)
print(k.describe())

# Top 10 frequently purchased products
print("\nTop 10 products:")
p=df.stack()
print(p.value_counts().head(10))


# %%

import matplotlib.pyplot as plt

# 1. Top 10 product frequencies
p.value_counts().head(10).plot(kind="bar")

plt.title("Top 10 Frequently Purchased Products")
plt.xlabel("Products")
plt.ylabel("Frequency")
plt.show()

# 2. Transaction-size distribution
k.plot(kind="hist", bins=15)

plt.title("Transaction Size Distribution")
plt.xlabel("Number of Products")
plt.ylabel("Number of Transactions")
plt.show()


# %%
from sklearn.preprocessing import MultiLabelBinarizer

transactions = df.apply(
    lambda row: row.dropna().tolist(), axis=1
).tolist()

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(transactions)

print("Feature matrix shape:", X.shape)
print("Unique products:", len(mlb.classes_))
print("Sample of encoded data:\n", X[:5])
print(mlb.classes_)


# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize the data
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Visualization")
plt.show()
X_pca.shape

# %%
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_pca )

print("Training clusters:", kmeans_labels)

# %%
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=3)
hc_labels = hc.fit_predict(X_pca )

print(hc_labels)

# %%
from sklearn.cluster import SpectralClustering,DBSCAN

sc = SpectralClustering(
    n_clusters=3,
    random_state=42,
    affinity="nearest_neighbors",
    n_neighbors=23
)

sc_labels = sc.fit_predict(X_pca )

print(sc_labels)

# %%
dbscan = DBSCAN(eps=0.1, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_pca)
print(dbscan_labels)

# %%
from sklearn.metrics import silhouette_score, davies_bouldin_score

# K-means
print("K-means:")
print("Silhouette Score:", silhouette_score(X_pca , kmeans_labels))
print("Davies-Bouldin Index:", davies_bouldin_score(X_pca , kmeans_labels))

# Hierarchical Clustering
print("\nHierarchical Clustering:")
print("Silhouette Score:", silhouette_score(X_pca , hc_labels))
print("Davies-Bouldin Index:", davies_bouldin_score(X_pca , hc_labels))

# Spectral Clustering
print("\nSpectral Clustering:")
print("Silhouette Score:", silhouette_score(X_pca , sc_labels))
print("Davies-Bouldin Index:", davies_bouldin_score(X_pca , sc_labels))

# DBSCAN Clustering
print("\nDBSCAN Clustering:")
print("Silhouette Score:", silhouette_score(X_pca , dbscan_labels))
print("Davies-Bouldin Index:", davies_bouldin_score(X_pca , dbscan_labels))


# %%



