import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.cluster import KMeans

df=pd.read_csv("star_with_gravity.csv")
df.head()

X=df.iloc[:,[3,4]].values
wcss=[]
for i in range(1,11): 
  kmeans=KMeans(n_clusters=i,init='k-means++',random_state=12)
  kmeans.fit(X)
  #wcss.append((kmeans.inertia_))
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel("number of clusters")
plt.show()