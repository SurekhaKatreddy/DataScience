K-Means:
In cluster analysis, the k-means algorithm can be used to partition the input data set into k partitions (clusters).
However, the pure k-means algorithm is not very flexible, and as such is of limited use (except for when vector quantization
as above is actually the desired use case). In particular, the parameter k is known to be hard to choose (as discussed above) 
when not given by external constraints. Another limitation is that it cannot be used with arbitrary distance functions or on 
non-numerical data. For these use cases, many other algorithms are superior.

Input: k = number of clusters to be created.
Output: The data points are clustered into k clusters

# Application of KMeans
1. document clustering
2. identifying crime-prone areas
3. customer segmentation
4. insurance fraud detection
5. public transport data analysis
6. clustering of IT alerts
