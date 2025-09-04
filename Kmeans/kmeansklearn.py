import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

## Step 1: Create a Custom Dataset

data = {
    'IQ Score': [105, 120, 95, 115, 100, 130, 90, 110, 108, 98, 112, 125, 93],
    'Final Marks': [85, 95, 70, 90, 80, 98, 65, 88, 86, 75, 89, 92, 72]
}
df = pd.DataFrame(data)

print("Step 1: Dataset Created")
print(df)

X = df.iloc[:, :].values

##  Step 2: Use the Elbow Method to Find the Optimal 'k'

wcss = []
for i in range(1, 8):
    kmeans_model = KMeans(n_clusters=i, init='k-means++', max_iter=50, n_init=10, random_state=45)
    kmeans_model.fit(X)
    wcss.append(kmeans_model.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 8), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Based on the plot, the optimal k is likely 3
optimal_k = 3


## Step 3: Train the K-means Model with Optimal 'k'

kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

print("\nModel Trained with Optimal K = 3")
print("Final Centroids:")
print(centroids)


df['Cluster'] = y_kmeans
print("\nFinal Cluster Assignments:")
print(df)

# ...existing code...

## Step 4: Visualize the Clusters
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green']
labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']

for i in range(optimal_k):
    # Plot points for each cluster
    cluster_points = X[y_kmeans == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
               c=colors[i], label=labels[i])

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], 
           c='yellow', marker='*', s=200, label='Centroids')

plt.title('Student Clusters based on IQ Score and Final Marks')
plt.xlabel('IQ Score')
plt.ylabel('Final Marks')
plt.legend()
plt.grid(True)
plt.show()