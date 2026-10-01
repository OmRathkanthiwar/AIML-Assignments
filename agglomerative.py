import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# LOAD DATASET
df = pd.read_csv("Mall_Customers.csv")

print(df.head())
print(df.isnull().sum())

# SELECT FEATURES
X = df[[
    "Annual Income (k$)",
    "Spending Score (1-100)"
]].values

# DRAW DENDROGRAM
plt.figure(figsize=(10, 6))

dendrogram(
    linkage(X, method="ward")
)

plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

# HIERARCHICAL CLUSTERING
model = AgglomerativeClustering(
    n_clusters=5,
    metric="euclidean",
    linkage="ward"
)

yPred = model.fit_predict(X)

# SILHOUETTE SCORE
score = silhouette_score(X, yPred)

print("Silhouette Score:", score)

# VISUALIZE CLUSTERS
plt.scatter(
    X[yPred == 0, 0],
    X[yPred == 0, 1],
    label="Cluster 1"
)

plt.scatter(
    X[yPred == 1, 0],
    X[yPred == 1, 1],
    label="Cluster 2"
)

plt.scatter(
    X[yPred == 2, 0],
    X[yPred == 2, 1],
    label="Cluster 3"
)

plt.scatter(
    X[yPred == 3, 0],
    X[yPred == 3, 1],
    label="Cluster 4"
)

plt.scatter(
    X[yPred == 4, 0],
    X[yPred == 4, 1],
    label="Cluster 5"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using Hierarchical Clustering")
plt.legend()
plt.show()