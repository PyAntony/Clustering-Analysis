# Clustering-Analysis

The work found in the **clustering.pdf** document is an overview on the K-Means algorithm. Contents include:

- Definition and algorithm description.
- Application in Python and R.
- How to select the number of clusters: the elbow method.
- Testing cluster quality using Silhouette Analysis.

In addition there is a full application of the algorithm on a Walmart dataset provided by Kaggle. It is a walk-trough of a proper machine learning exercise including data preparation, data exploration, algorithm implementation and analysis. You can find this walk-trough in the **walmart.ipynb** notebook. Additional notebooks used can be found in the *notebooks* folder.

## Extracts 

### The K-means Algorithm – Definition (from clustering.pdf)

This simple algorithm is commonly used to create clusters by proximity. It is easy to implement
and computationally faster than others algorithms in the same class. It belongs to a group called
Prototype-based clustering since there is a prototype that represents the cluster, either the average
(centroid) or the most frequent (medoid) item.  
This is a visual representation of 150 data points grouped together by K-means. Each color
represents a cluster. There is no specific order (hierarchy) or labels applied to clusters:

![pic1]()

How are data points classified? It is important to note that this function doesn’t come up with
the best number of classes (k); we must specify the number of classes we want beforehand. With that
value, the function compute the following 4 steps:

a) Every single data point is randomly assigned to a cluster.

![pic2]()

b) The cluster centroids -the mean of the data points in the clusters- are found.

![pic3]()

c) Data points are assigned to the same cluster as the closest centroid (by squared Euclidean distance).

![pic4]()

d) Steps b and c will iterate until the function finds no more changes in the clusters.

![pic5]()

K-means can also be described as an optimization function where the goal is to minimize the
total distance between the centroids and the data points around them (local distance). The objective
formula is:

![pic6]()








	


     



 
