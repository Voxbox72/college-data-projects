import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("train.csv").dropna()
pd.set_option('display.max_rows',0)


#plotting crimes using coordinates to get an idea of the distribution
pdf = df[['X','Y']]
fig = plt.figure()
plt.title('SF Crime Distribution')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.scatter(pdf.X, pdf.Y, s = 2, c = 'r')
plt.show()

#Finding how many outliers 
outlier90 = df[df["Y"] == 90]
print(outlier90.count())


#Catching the outlier crimes and removing it 
df_f = df.query('Y != 90')


#Plotting the actual distribution without the outliers
pdf = df_f[['X','Y']]
fig = plt.figure()
plt.title('SF Crime Distribution')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.scatter(pdf.X, pdf.Y, s = 2, c = 'r')
plt.show()




#Data is too big so we take part of the data to reduce computation time 
df_f = df_f.head(10000)





#Combining Coordinates to make them a single feature
df_coor = pd.DataFrame({'Longitude': df_f["X"], 'Latitude': df_f["Y"]})



#Find optimal number of clusters
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_coor)
    silhouette_scores.append(silhouette_score(df_coor, kmeans.labels_))




#Plotting results from silhouette score testing    
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Kmeans')
plt.show()


# Fit a KMeans model with 3 clusters
model = KMeans(n_clusters=3).fit(df_coor)

# add the cluster labels to the original DataFrame
df_clusters = model.labels_
df_f['cluster_label'] = df_clusters

# convert the cluster labels to a pandas DataFrame
cluster_df = pd.DataFrame(df_clusters)
print(df_f.head(10))

# Calculate the silhouette score
silhouette_avg = silhouette_score(df_coor, model.labels_)


centers = model.cluster_centers_
print("Kmeans cluster centers:",centers)
long_centers = [i for i, _ in centers]
lat_centers = [i for _, i in centers]


# Plotting the cluster labels

plt.scatter(df_f["X"], df_f["Y"], c=df_f["cluster_label"])
plt.scatter(long_centers,lat_centers, s=50, c='r', marker='+')
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("San Francisco")

#Plot cluster centers 
plt.scatter(df_f["X"], df_f["Y"], c=df_f["cluster_label"])
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=50, c='r', marker='+')
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("San Francisco Crime Clusters Using Kmeans")
plt.show()

print("Silhouette Score using Kmeans:", silhouette_avg)
