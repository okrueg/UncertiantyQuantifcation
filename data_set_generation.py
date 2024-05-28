import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification

#def exclusionArea(x: ,y):

rand = 20
x, y = make_classification(n_samples= 100,
                            n_features=2,
                            n_redundant=0,
                            n_informative=2, 
                            n_clusters_per_class=1,
                            class_sep= 2.0,
                            random_state= rand
)
print(type(x))
plt.scatter(x[:,0], x[:,1],c=y)
plt.show()