import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv("train.csv").dropna()
pd.set_option('display.max_rows', 0)


# Catching the outlier crimes and removing them
df_f = df.query('Y != 90')


# Data is too big, so we take a part of the data to reduce computation time
df_f = df_f.head(10000)

# Combining Coordinates to make them a single feature
df_coor = pd.DataFrame({'Longitude': df_f["X"], 'Latitude': df_f["Y"]})

# Find optimal number of clusters
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    kmedoids = KMedoids(n_clusters=k, random_state=42)
    kmedoids.fit(df_coor)
    silhouette_scores.append(silhouette_score(df_coor, kmedoids.labels_))

# Plotting results from silhouette score testing
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Kmedoids')
plt.show()

# Fit a KMedoids model with 3 clusters
model = KMedoids(n_clusters=3, random_state=42).fit(df_coor)

# Add the cluster labels to the original DataFrame
df_f['cluster_label'] = model.labels_

#Showing cluster centers
centers = model.cluster_centers_
print("Kmedoids cluster centers:",centers)

# Plotting the cluster labels
plt.scatter(df_f["X"], df_f["Y"], c=df_f["cluster_label"])
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=50, c='r', marker='+')
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("San Francisco Crime Clusters using Kmedoids")
plt.show()

# Calculate the silhouette score
silhouette_avg = silhouette_score(df_coor, model.labels_)
print("Silhouette Score using Kmedoids:", silhouette_avg)
