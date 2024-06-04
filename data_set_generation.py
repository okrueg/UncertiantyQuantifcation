import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_moons, make_blobs

#--- Returns values outside of a circle ---#
def exclusion_area(x: np.ndarray, y: np.ndarray, user_points: np.ndarray, radius: int, position: np.ndarray  ):
    distance = np.sqrt((x[:,0] - position[0])**2 + (x[:,1] - position[1])**2)

    include_points = user_points == 1
    outside_radius = distance >= radius

    kept_indexs = np.where(outside_radius | include_points)
    
    # Return the kept indices or the corresponding points from x array
    filtered_x = x[kept_indexs]
    filtered_y = y[kept_indexs]
    return filtered_x, filtered_y

#--- Utilizes matplotlib to plot *Outdated* ---
def plot_data_static(x: np.ndarray, y: np.ndarray, circle = None):
    figure, axes = plt.subplots()
    #axes.add_artist( circle )
    axes.scatter(x[:,0], x[:,1],c=y)
    axes.set_ylim(-5,5)
    axes.set_xlim(-5,5)
    plt.show()

#-- Just a default tuned generation --#
def generate_data(num_points: int, type = 'blobs'):
    x,y = None, None
    if type == 'classification':
        x, y = make_classification(n_samples= num_points,
                                    n_features=2,
                                    n_redundant=0,
                                    n_informative=2, 
                                    n_clusters_per_class=1,
                                    flip_y= 0,
                                    class_sep= 1.2,
                                    #random_state= rand
                                    )
    elif type == 'moons':
        x,y = make_moons(n_samples=num_points,
                         noise=0.1)
    elif type == 'blobs':
        x,y = make_blobs(n_samples=num_points,
                        n_features=2,
                        centers= [[-1,1],[1,-1]],
                        cluster_std= 0.35)
        
    user_points = np.zeros(num_points, dtype=np.int8)

    return x,y, user_points

def generate_shifted(num_points: int, shift= [0,0], type = 'blobs'):
    x,y = None, None
    centers = np.array([1+shift[0], -1+shift[1]])
    centers = centers.reshape(1,-1)
    if type == 'blobs':
        x,y = make_blobs(n_samples=num_points,
                        n_features=2,
                        centers= centers,
                        cluster_std= 0.1)
    user_points = np.ones(num_points, dtype=np.int8)
    return x,y, user_points