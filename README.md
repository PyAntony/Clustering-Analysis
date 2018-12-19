# Clustering-Analysis

The work found in the **clustering.pdf** document is an overview on the K-Means algorithm. Contents include:

- Definition and algorithm description.
- Application in Python and R.
- How to select the number of clusters: the elbow method.
- Testing cluster quality using Silhouette Analysis.

In addition there is a full implementation of the algorithm on a Walmart dataset provided by Kaggle. It is a walk-trough of a proper machine learning exercise including data preparation, data exploration, algorithm implementation and analysis. You can find this walk-trough in the **walmart.ipynb** notebook. Additional notebooks used can be found in the *Notebooks* folder.

## Extracts 

### The K-means Algorithm – Definition (from clustering.pdf)

This simple algorithm is commonly used to create clusters by proximity. It is easy to implement
and computationally faster than others algorithms in the same class. It belongs to a group called
Prototype-based clustering since there is a prototype that represents the cluster, either the average
(centroid) or the most frequent (medoid) item.  
This is a visual representation of 150 data points grouped together by K-means. Each color
represents a cluster. There is no specific order (hierarchy) or labels applied to clusters:

![pic1](https://github.com/PyAntony/Clustering-Analysis/blob/master/images/pic1.png)

How are data points classified? It is important to note that this function doesn’t come up with
the best number of classes (k); we must specify the number of classes we want beforehand. With that
value, the function compute the following 4 steps:

a) Every single data point is randomly assigned to a cluster.

![pic2](https://github.com/PyAntony/Clustering-Analysis/blob/master/images/pic2.png)

b) The cluster centroids -the mean of the data points in the clusters- are found.

![pic3](https://github.com/PyAntony/Clustering-Analysis/blob/master/images/pic3.png)

c) Data points are assigned to the same cluster as the closest centroid (by squared Euclidean distance).

![pic4](https://github.com/PyAntony/Clustering-Analysis/blob/master/images/pic4.png)

d) Steps b and c will iterate until the function finds no more changes in the clusters.

![pic5](https://github.com/PyAntony/Clustering-Analysis/blob/master/images/pic5.png)

K-means can also be described as an optimization function where the goal is to minimize the
total distance between the centroids and the data points around them (local distance). The objective
formula is:

![pic6](https://github.com/PyAntony/Clustering-Analysis/blob/master/images/pic6.png)
<img src="https://github.com/PyAntony/Clustering-Analysis/blob/master/images/pic6.png" width="100" height="100" />

### Silhouette Analysis on Walmart dataset (walmart.ipynb)


```python
# Let's explore how the Silhouette Plot performs for different values of k 
from sklearn.metrics import silhouette_samples

# to generate 14 models for k in range (2,16)
m = []
for k in range(2,16):
    model = KMeans(n_clusters=k, random_state=0)
    m.append(model.fit(raw))

clusters = [c.labels_ for c in m]

# to generate the coefficients
coefs = [silhouette_samples(raw,i) for i in clusters]
# to set the number of labels and clusters
unique_labels_list = [np.unique(i) for i in clusters]
clusters_total_num = [i.shape[0] for i in unique_labels_list]

from matplotlib import cm

# loop to generate 14 plots starting from k=2
for i in range(0,14):
    col=.01
    y_low,y_up=0,0 
    ticks =[]
    plt.figure(figsize=(6,5), facecolor='lightgray',frameon=True)
    plt.style.use('seaborn-dark-palette')
    plt.title('For k='+str(i+2))
    for b in unique_labels_list[i]:        
        val_c = coefs[i][clusters[i]==b]
        val_c.sort()
        y_up += len(val_c)
        cl =cm.Set1(X=col)        
        #plot
        plt.barh(bottom=range(y_low, y_up), 
                width=val_c, 
                height=1.2,
                edgecolor=cl)
        ticks.append((y_low+y_up)/2)
        y_low += len(val_c)
        col += .04

    #to get the coefficients average and plot it as a line    
    coef_avg = np.mean(coefs[i]) 
    plt.axvline(coef_avg, color="black", ls='--', lw=3)

    #to modify the ticks
    plt.yticks(ticks, unique_labels_list[i]+1)
    plt.ylabel('CLUSTERS')
    plt.xlabel('COEFFICIENTS')
    plt.tight_layout()
```

![pic7]()




	


     



 
