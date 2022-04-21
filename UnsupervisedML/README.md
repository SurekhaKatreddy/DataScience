Unsupervised learning, also known as unsupervised machine learning, uses machine learning algorithms to analyze and cluster unlabeled datasets. 
These algorithms discover hidden patterns or data groupings without the need for human intervention. Its ability to discover similarities and 
differences in information make it the ideal solution for exploratory data analysis, cross-selling strategies, customer segmentation, and image 
recognition.

# Common Approaches

1. Clustering - a data mining technique which groups unlabeled data based on their similarities or differences. Ex: KMeans
2. Hierarchical clustering, also known as hierarchical cluster analysis (HCA) can be agglomerative or divisive. 

   Divisive approach : top-down approach in which data cluster is divided based on the differences between data points.
 
   Four different methods are commonly used to measure similarity:
   Ward’s linkage: This method states that the distance between two clusters is defined by the increase in the sum of squared after the clusters are merged.
   Average linkage: This method is defined by the mean distance between two points in each cluster
   Complete (or maximum) linkage: This method is defined by the maximum distance between two points in each cluster
   Single (or minimum) linkage: This method is defined by the minimum distance between two points in each cluster
   
3. probabilistic clustering - data points are clustered based on the likelihood that they belong to a particular distribution.
   GGaussian Mixture Models are classified as mixture models, which means that they are made up of an unspecified number of probability distribution 
   functions. GMMs are primarily leveraged to determine which Gaussian, or normal, probability distribution a given data point belongs to. If the mean 
   or variance are known, then we can determine which distribution a given data point belongs to. However, in GMMs, these variables are not known, 
   so we assume that a latent, or hidden, variable exists to cluster data points appropriately. While it is not required to use the 
   Expectation-Maximization (EM) algorithm, it is a commonly used to estimate the assignment probabilities for a given data point to a particular 
   data cluster.
   
4. Association rules - ule-based method for finding relationships between variables in a given dataset. These methods are frequently used for market
   basket analysis, allowing companies to better understand relationships between different products.    
   
