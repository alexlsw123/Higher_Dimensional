import numpy as np
from pyspark.ml.linalg import Vectors
from pyspark.conf import SparkConf
import pyspark.sql.functions as func
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans 
from pyspark.ml.evaluation import ClusteringEvaluator 
import plotly.express as px
import plotly.io as pio
import math


data = sc.textFile("space.dat")
data_split = data.map(lambda x:x.split(","))
data_float = data_split.map(lambda x: [float(x[0]), float(x[1]), 
float(x[2]), float(x[3]), float(x[4]), float(x[5])])

# Making into data frame (spark)
rdd_df = spark.createDataFrame(data_float, ['a', 'b', 'c', 'd', 'e', 'f'])

# Visualization code (extracting data to Jupyter lab for visualization)
assembler = VectorAssembler(inputCols = rdd_df.columns, outputCol = 
"points")

data_df = assembler.transform(rdd_df).select("points")
 
data_df = data_df.rdd.map(lambda row: row.points).collect()

np.array(data_df).tofile("data.out")

# Removing outliers (found 4 outliers we can remove from visualization)
rdd_df_clean = rdd_df.where((func.col("a") != 11) & (func.col("a") != 91)) 
# the points where first column contained 11 and 91 was found outliers by 
observing the plot—attaching the plot)

# Clustering
Vec_assembler = VectorAssembler(inputCols = rdd_df_clean.columns, 
outputCol='features')

final_data = Vec_assembler.transform(rdd_df_clean)
final_data.select('features').show(5)

# Finding optimal number of clusters

silhouette_score=[]

evaluator = ClusteringEvaluator(featuresCol = 'features',
                                 metricName = 'silhouette',
                                 distanceMeasure ='squaredEuclidean')

for i in range(2, 10):
     kmeans = KMeans(featuresCol = 'features',  k = i)
     model = kmeans.fit(final_data)
     predictions = model.transform(final_data)
     score = evaluator.evaluate(predictions)
     silhouette_score.append(score)
     print('Silhouette Score for k =', i, 'is', score) # Optimal number of 
cluster is 6

# Finding center of each clusters
kmeans = KMeans(featuresCol = 'features', k = 6) 
model = kmeans.fit(final_data) 
predictions = model.transform(final_data)

centers = model.clusterCenters() 
print("Cluster Centers: ") 
for center in centers: 
    print(center)


# Visualization and PCA for each clusters
 
assembler_c = VectorAssembler(inputCols = ['a', 'b', 'c', 'd', 'e', 'f'], 
outputCol = 'values')

# Cluster 1 
# No-PCA visualization
cluster1_df = predictions.where(func.col("prediction") == 0)

cluster1_df = assembler_c.transform(cluster1_df).select('values')

cluster1 = cluster1_df.rdd.map(lambda row: row.values).collect()

clust1_arr = np.array(cluster1)

points_clust1 = len(clust1_arr)

c1_3d = px.scatter_3d(clust1_arr, x = 0, y = 1, z = 2)
c1_3d.update_traces(marker = dict(size = 1.5, color = "red"))
c1_3d.update_layout(title_text = "Cluster 1 plot", scene = 
dict(xaxis_title = 'Column 1', yaxis_title = 'Column 2', zaxis_title = 
'Column 3'))
pio.write_image(c1_3d, 
"/jet/home/sleen/Spark/38614_HW4/sparkplot/cluster1_plot.png")

# PCA

scaler_c1 = StandardScaler(
     inputCol = 'values',
     outputCol = 'scaledFeatures',
     withMean = True,
     withStd = False
 ).fit(cluster1_df)
 
df_scaled_c1 = scaler_c1.transform(cluster1_df)

n_components = 6
pca_c1 = PCA(
     k = n_components,
     inputCol = 'scaledFeatures',
     outputCol = 'PCAfeatures'
).fit(df_scaled_c1)

df_pca_c1 = pca_c1.transform(df_scaled_c1)
 
X_pca_c1 = df_pca_c1.rdd.map(lambda row: row.PCAfeatures).collect()
X_pca_c1 = np.array(X_pca_c1)

print(pca_c1.explainedVariance) # True dimension for cluster 1 object

pca_c1_3d = px.scatter_3d(X_pca_c1, x = 0, y = 1, z = 2)
pca_c1_3d.update_traces(marker = dict(size = 1.5, color = "red"))
pca_c1_3d.update_layout(title_text = "Cluster 1 PCA plot", scene = 
dict(xaxis_title = 'PC1', yaxis_title = 'PC2', zaxis_title = 'PC3'))
pio.write_image(pca_c1_3d, 
"/jet/home/sleen/Spark/38614_HW4/sparkplot/cluster1_PCA_plot.png")


# Finding Length, Width for Cluster 1

cluster1_max = []

for i in range(0, 6):
     cluster1_max.append(max(clust1_arr[:, i]))

cluster1_max = np.array(cluster1_max)

cluster1_center = centers[0]

center_to_max1 = abs(cluster1_max - cluster1_center)

length_cube = np.mean(center_to_max1) * 2

# Cluster 2
# No-PCA visualization
cluster2_df = predictions.where(func.col("prediction") == 1)

cluster2_df = assembler_c.transform(cluster2_df).select('values')

cluster2 = cluster2_df.rdd.map(lambda row: row.values).collect()

clust2_arr = np.array(cluster2)

points_clust2 = len(clust2_arr)

c2_3d = px.scatter_3d(clust2_arr , x = 1, y = 4, z = 5)
c2_3d.update_traces(marker = dict(size = 3, color = "red"))
c2_3d.update_layout(title_text = "Cluster 2 plot", scene = 
dict(xaxis_title = 'Column 1', yaxis_title = 'Column 2', zaxis_title = 
'Column 3'))
pio.write_image(c2_3d, 
"/jet/home/sleen/Spark/38614_HW4/sparkplot/cluster2_plot.png")

# PCA

scaler_c2 = StandardScaler(
     inputCol = 'values',
     outputCol = 'scaledFeatures',
     withMean = True,
     withStd = False
 ).fit(cluster2_df)
 
df_scaled_c2 = scaler_c2.transform(cluster2_df)

n_components = 6
pca_c2 = PCA(
     k = n_components,
     inputCol = 'scaledFeatures',
     outputCol = 'PCAfeatures'
).fit(df_scaled_c2)

df_pca_c2 = pca_c2.transform(df_scaled_c2)
 
X_pca_c2 = df_pca_c2.rdd.map(lambda row: row.PCAfeatures).collect()
X_pca_c2 = np.array(X_pca_c2)

print(pca_c2.explainedVariance) # True dimension for cluster 2 object

pca_c2_3d = px.scatter_3d(X_pca_c2, x = 1, y = 4, z = 5)
pca_c2_3d.update_traces(marker = dict(size = 3, color = "red"))
pca_c2_3d.update_layout(title_text = "Cluster 2 PCA plot", scene = 
dict(xaxis_title = 'PC1', yaxis_title = 'PC2', zaxis_title = 'PC3'))
pio.write_image(pca_c2_3d, 
"/jet/home/sleen/Spark/38614_HW4/sparkplot/cluster2_PCA_plot.png")


# Finding Length, Width for Cluster 2

cluster2_max = []

for i in range(0, 6):
     cluster2_max.append(max(clust2_arr[:, i]))

cluster2_max = np.array(cluster2_max)

cluster2_min = []

for i in range(0, 6):
      cluster2_min.append(min(clust2_arr[:, i]))

cluster2_min = np.array(cluster2_min)

length_vec = abs(cluster2_max - cluster2_min)

# Cluster 3
# No-PCA visualization
cluster3_df = predictions.where(func.col("prediction") == 2)

cluster3_df = assembler_c.transform(cluster3_df).select('values')

cluster3 = cluster3_df.rdd.map(lambda row: row.values).collect()

clust3_arr = np.array(cluster3)

points_clust3 = len(clust3_arr)

c3_2d = px.scatter(clust3_arr , x = 0, y = 1)
c3_2d.update_traces(marker = dict(size = 5, color = "red"))
c3_2d.update_layout(title_text = "Cluster 3 plot", scene = 
dict(xaxis_title = 'Column 1', yaxis_title = 'Column 2'))
pio.write_image(c3_2d, 
"/jet/home/sleen/Spark/38614_HW4/sparkplot/cluster3_plot.png")

# PCA

scaler_c3 = StandardScaler(
     inputCol = 'values',
     outputCol = 'scaledFeatures',
     withMean = True,
     withStd = False
 ).fit(cluster3_df)
 
df_scaled_c3 = scaler_c3.transform(cluster3_df)

n_components = 6
pca_c3 = PCA(
     k = n_components,
     inputCol = 'scaledFeatures',
     outputCol = 'PCAfeatures'
).fit(df_scaled_c3)

df_pca_c3 = pca_c3.transform(df_scaled_c3)
 
X_pca_c3 = df_pca_c3.rdd.map(lambda row: row.PCAfeatures).collect()
X_pca_c3 = np.array(X_pca_c3)

print(pca_c3.explainedVariance) # True dimension for cluster 3 object

pca_c3_2d = px.scatter(X_pca_c3, x = 0, y = 1)
pca_c3_2d.update_traces(marker = dict(size = 5, color = "red"))
pca_c3_2d.update_layout(title_text = "Cluster 3 PCA plot", scene = 
dict(xaxis_title = 'PC1', yaxis_title = 'PC2'))
pio.write_image(pca_c3_2d, 
"/jet/home/sleen/Spark/38614_HW4/sparkplot/cluster3_PCA_plot.png")

# Finding Diagonals for Cluster 3

length_rec = abs(max(X_pca_c3[:, 0] - min(X_pca_c3[:, 0])))
width_rec = abs(max(X_pca_c3[:, 1] - min(X_pca_c3[:, 1])))

# Cluster 4 
# No-PCA visualization
cluster4_df = predictions.where(func.col("prediction") == 3)

cluster4_df = assembler_c.transform(cluster4_df).select('values')

cluster4 = cluster4_df.rdd.map(lambda row: row.values).collect()

clust4_arr = np.array(cluster4)

points_clust4 = len(clust4_arr)

c4_3d = px.scatter_3d(clust4_arr , x = 0, y = 1, z = 2)
c4_3d.update_traces(marker = dict(size = 1.5, color = "red"))
c4_3d.update_layout(title_text = "Cluster 4 plot", scene = 
dict(xaxis_title = 'Column 1', yaxis_title = 'Column 2', zaxis_title = 
'Column 3'))
pio.write_image(c4_3d, 
"/jet/home/sleen/Spark/38614_HW4/sparkplot/cluster4_plot.png")

# PCA

scaler_c4 = StandardScaler(
     inputCol = 'values',
     outputCol = 'scaledFeatures',
     withMean = True,
     withStd = False
 ).fit(cluster4_df)
 
df_scaled_c4 = scaler_c4.transform(cluster4_df)

n_components = 6
pca_c4 = PCA(
     k = n_components,
     inputCol = 'scaledFeatures',
     outputCol = 'PCAfeatures'
).fit(df_scaled_c4)

df_pca_c4 = pca_c4.transform(df_scaled_c4)
 
X_pca_c4 = df_pca_c4.rdd.map(lambda row: row.PCAfeatures).collect()
X_pca_c4 = np.array(X_pca_c4)

print(pca_c4.explainedVariance) # True dimension for cluster 4 object

pca_c4_3d = px.scatter_3d(X_pca_c4, x = 1, y = 4, z = 5)
pca_c4_3d.update_traces(marker = dict(size = 1.5, color = "red"))
pca_c4_3d.update_layout(title_text = "Cluster 4 PCA plot", scene = 
dict(xaxis_title = 'PC1', yaxis_title = 'PC2', zaxis_title = 'PC3'))
pio.write_image(pca_c4_3d, 
"/jet/home/sleen/Spark/38614_HW4/sparkplot/cluster4_PCA_plot.png")

# Finding Length, Width for Cluster 4

height_cuboid = max(X_pca_c4[:, 0]) - min(X_pca_c4[:, 0])
lengthwidth_cuboid = math.sqrt((1/2) * (abs(max(X_pca_c4[:, 1]) - 
min(X_pca_c4[:, 1])) ** 2))


# Cluster 5
# No-PCA visualization
cluster5_df = predictions.where(func.col("prediction") == 4)

cluster5_df = assembler_c.transform(cluster5_df).select('values')

cluster5 = cluster5_df.rdd.map(lambda row: row.values).collect()

clust5_arr = np.array(cluster5)

points_clust5 = len(clust5_arr)

c5_2d = px.scatter(clust5_arr , x = 0, y = 1)
c5_2d.update_traces(marker = dict(size = 5, color = "red"))
c5_2d.update_layout(title_text = "Cluster 5 plot", scene = 
dict(xaxis_title = 'Column 1', yaxis_title = 'Column 2'))
pio.write_image(c5_2d, 
"/jet/home/sleen/Spark/38614_HW4/sparkplot/cluster5_plot.png")

# PCA

scaler_c5 = StandardScaler(
     inputCol = 'values',
     outputCol = 'scaledFeatures',
     withMean = True,
     withStd = False
 ).fit(cluster5_df)
 
df_scaled_c5 = scaler_c5.transform(cluster5_df)

n_components = 6
pca_c5 = PCA(
     k = n_components,
     inputCol = 'scaledFeatures',
     outputCol = 'PCAfeatures'
).fit(df_scaled_c5)

df_pca_c5 = pca_c5.transform(df_scaled_c5)
 
X_pca_c5 = df_pca_c5.rdd.map(lambda row: row.PCAfeatures).collect()
X_pca_c5 = np.array(X_pca_c5)

print(pca_c5.explainedVariance) # True dimension for cluster 5 object

pca_c5_2d = px.scatter(X_pca_c5, x = 0, y = 1)
pca_c5_2d.update_traces(marker = dict(size = 5, color = "red"))
pca_c5_2d.update_layout(title_text = "Cluster 5 PCA plot", scene = 
dict(xaxis_title = 'PC1', yaxis_title = 'PC2'))
pio.write_image(pca_c5_2d, 
"/jet/home/sleen/Spark/38614_HW4/sparkplot/cluster5_PCA_plot.png")

# Finding Length, Width for Cluster 5

radius = abs(max(X_pca_c5[:, 0]) - min((X_pca_c5[:, 0]))) / 2

# Cluster 6
# No-PCA visualization
cluster6_df = predictions.where(func.col("prediction") == 5)

cluster6_df = assembler_c.transform(cluster6_df).select('values')

cluster6 = cluster6_df.rdd.map(lambda row: row.values).collect()

clust6_arr = np.array(cluster6)

points_clust6 = len(clust6_arr)

c6_2d = px.scatter(clust6_arr , x = 0, y = 1)
c6_2d.update_traces(marker = dict(size = 5, color = "red"))
c6_2d.update_layout(title_text = "Cluster 6 plot", scene = 
dict(xaxis_title = 'Column 1', yaxis_title = 'Column 2'))
pio.write_image(c6_2d, 
"/jet/home/sleen/Spark/38614_HW4/sparkplot/cluster6_plot.png")

# PCA

scaler_c6 = StandardScaler(
     inputCol = 'values',
     outputCol = 'scaledFeatures',
     withMean = True,
     withStd = False
 ).fit(cluster6_df)
 
df_scaled_c6 = scaler_c6.transform(cluster6_df)

n_components = 6
pca_c6 = PCA(
     k = n_components,
     inputCol = 'scaledFeatures',
     outputCol = 'PCAfeatures'
).fit(df_scaled_c6)

df_pca_c6 = pca_c6.transform(df_scaled_c6)
 
X_pca_c6 = df_pca_c6.rdd.map(lambda row: row.PCAfeatures).collect()
X_pca_c6 = np.array(X_pca_c6)

print(pca_c6.explainedVariance) # True dimension for cluster 6 object

pca_c6_2d = px.scatter(X_pca_c6, x = 0, y = 1)
pca_c6_2d.update_traces(marker = dict(size = 5, color = "red"))
pca_c6_2d.update_layout(title_text = "Cluster 6 PCA plot", scene = 
dict(xaxis_title = 'PC1', yaxis_title = 'PC2'))
pio.write_image(pca_c6_2d, 
"/jet/home/sleen/Spark/38614_HW4/sparkplot/cluster6_PCA_plot.png")

# Finding Length, Width for Cluster 6

length_line = abs(max(X_pca_c6[:, 0] - min(X_pca_c6[:, 0])))≈
