import numpy as np
import pandas as pd
import mean_shift as ms
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from sklearn.cluster import MeanShift, estimate_bandwidth



#df = pd.read_csv("handwritten/semeion.data", delimiter=r"\s+",
 #                header=None)

#X = pd.DataFrame(df)
#X = X.drop([256, 257, 258, 259, 260, 261, 262, 263, 264, 265], axis=1)
#label_df = pd.DataFrame(df.iloc[:, [256, 257, 258, 259, 260, 261, 262, 263, 264, 265]])
#label_df.rename(columns={256: 0, 257: 1, 258: 2, 259: 3, 260: 4, 261: 5, 262: 6, 263: 7, 264: 8, 265: 9},
 #               inplace=True)
#y = label_df

#print("yy = ", y.shape)
#print("x = ", X.shape)

import mean_shift as ms

#data = pd.read_csv("handwritten/semeion.data", delimiter=r"\s+",
 #                header=None)
data = np.genfromtxt('handwritten/semeion.csv', delimiter=',')
#mean_shifter = ms.MeanShift()
#mean_shift_result = mean_shifter.cluster(data, kernel_bandwidth = 10)

mean_shifter = ms.MeanShift(kernel='multivariate_gaussian')
mean_shift_result = mean_shifter.cluster(data, kernel_bandwidth = [10,20,30,40,50,60,70,80,90,10,20,30,40,50,60,70,80,90
                                                                   ,10,20,30,40,50,60,70,80,90,10,20,30,40,50,60,70,80,90
                                                                   ,10,20,30,40,50,60,70,80,90,10,20,30,40,50,60,70,80,90
                                                                   ,10,20,30,40,50,60,70,80,90,10,20,30,40,50,60,70,80,90
                                                                   ,10,20,30,40,50,60,70,80,90,10,20,30,40,50,60,70,80,90
                                                                   ,10,20,30,40,50,60,70,80,90,10,20,30,40,50,60,70,80,90
                                                                   ,10,20,30,40,50,60,70,80,90,10,20,30,40,50,60,70,80,90
                                                                   ,10,20,30,40,50,60,70,80,90,10,20,30,40,50,60,70,80,90
                                                                   ,10,20,30,40,50,60,70,80,90,10,20,30,40,50,60,70,80,90
                                                                   ,10,20,30,40,50,60,70,80,90,10,20,30,40,50,60,70,80,90
                                                                   ,10,20,30,40,50,60,70,80,90,10,20,30,40,50,60,70,80,90
                                                                   ,10,20,30,40,50,60,70,80,90,10,20,30,40,50,60,70,80,90
                                                                   ,10,20,30,40,50,60,70,80,90,10,20,30,40,50,60,70,80,90
                                                                   ,10,20,30,40,50,60,70,80,90,10,20,30,40,50,60,70,80,90
                                                                   ,10,20,30,50
                                                                   ])


original_points =  mean_shift_result.original_points
shifted_points = mean_shift_result.shifted_points
cluster_assignments = mean_shift_result.cluster_ids
print(original_points)
print(shifted_points)
print(cluster_assignments)

x = original_points[:,0]
y = original_points[:,1]
Cluster = cluster_assignments
centers = shifted_points

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x,y,c=Cluster,s=1600)
for i,j in centers:
    ax.scatter(i,j,s=1600,c='red',marker='+')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(scatter)

fig.savefig("mean_shift_result")


