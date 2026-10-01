import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# LOAD DATASET
df = pd.read_csv("Mall_Customers.csv")

print(df.head())
print(df.isnull().sum())

# DROP CUSTOMER ID
data = df.drop("CustomerID", axis=1)

# LABEL ENCODING
encoder = LabelEncoder()
data["Genre"] = encoder.fit_transform(data["Genre"])

# FIND K USING ELBOW METHOD
inertia = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(data)
    inertia.append(model.inertia_)

# PLOT ELBOW GRAPH
plt.plot(range(1, 11), inertia)
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# CREATE K-MEANS MODEL
model = KMeans(n_clusters=5, random_state=42, n_init=10)
data["Labels"] = model.fit_predict(data)

# DISPLAY CLUSTERS
plt.scatter(
    data["Annual Income (k$)"],
    data["Spending Score (1-100)"],
    c=data["Labels"]
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.show()