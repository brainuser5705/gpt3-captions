import os
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LogNorm

from flask import session

def plot_data(data, color='k', size=2, marker='.'):
    """
    Assumes that data is a Pandas dataframes with two columns
    """
    plt.scatter(data.iloc[:,-2], data.iloc[:,-1], c=color, s=size, marker=marker)
    plt.title(session['title'])
    plt.xlabel(session['x-axis'])
    plt.ylabel(session['y-axis'])


def create_plot_img(fig_name, fig_extension):
    img_name = fig_name + '.' + fig_extension
    plt.savefig(os.path.join('static', img_name), format=fig_extension, dpi=300)
    return img_name


def plot_centroids(centroids, weights=None):

    if weights is not None:
        centroids = centroids[weights > weights.max()/10]

     # plot the white circle of centroid markers
    plt.scatter(centroids[:,0], centroids[:,1],
                marker='o', s=35, linewidths=8,
                color='w', zorder=10, alpha=0.9)
    
    # plot the x of the centroid markers
    plt.scatter(centroids[:,0], centroids[:,1],
                marker='x', s=2, linewidths=12,
                color='k', zorder=11, alpha=1)


def plot_decision_boundaries(clusterer, data):
    
    # Get the extremas of the dataset
    mins = data.min(axis=0) - 0.1
    maxs = data.max(axis=0) + 0.1
    
    resolution = 1000
    # Create a meshgrid of using extremas as a boundary
    # Shape will be resolution X resolution
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    
    # Predict all datapoints within points generated by meshgrid
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # plot filled contours
    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                 cmap="Pastel2")
    
    # plot lines
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')


def plot_gaussian_mixture(clusterer, data):
    mins = data.min(axis=0) - 0.1
    maxs = data.max(axis=0) + 0.1
    
    resolution = 1000
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 norm=LogNorm(vmin=1.0, vmax=30.0),
                 levels=np.logspace(0, 2, 12))
    plt.contour(xx, yy, Z,
                norm=LogNorm(vmin=1.0, vmax=30.0),
                levels=np.logspace(0, 2, 12),
                linewidths=1, colors='k')

    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z,
                linewidths=2, colors='r', linestyles='dashed')
    
    