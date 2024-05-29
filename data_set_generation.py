import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_moons

#--- Returns values outside of a circle ---#
def exclusion_area(x: np.ndarray, y: np.ndarray, radius: int, position = np.ndarray ):
    distance = np.sqrt((x[:,0] - position[0])**2 + (x[:,1] - position[1])**2)
    kept_indexs = np.where(distance >= radius)
    filtered_x = x[kept_indexs]
    filtered_y = y[kept_indexs]
    return filtered_x, filtered_y

#--- Utilizes matplotlib to plot *Outdated* ---
def plot_data_static(x: np.ndarray, y: np.ndarray, circle = None):
    figure, axes = plt.subplots()
    axes.add_artist( circle )
    axes.scatter(x[:,0], x[:,1],c=y)
    axes.set_ylim(-5,5)
    axes.set_xlim(-5,5)
    plt.show()

#-- Just a default tuned generation --#
def generate_data():
    x, y = make_classification(n_samples= 100,
                                n_features=2,
                                n_redundant=0,
                                n_informative=2, 
                                n_clusters_per_class=2,
                                flip_y= 0,
                                class_sep= 1.2,
                                #random_state= rand
                            )
    return x,y