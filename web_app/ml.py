import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import plotting

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# To make consistent results for the same uploaded dataset
np.random.seed(42)

def process_data(data):
    x = data[data.columns[-2]]
    y = data[data.columns[-1]]
    return x, y

def regression(data):

    x, y = process_data(data)

    # X must be reshaped into a 2d array
    x_2d = x.to_numpy().reshape(x.shape[0], 1)
    lin_reg = LinearRegression()
    lin_reg.fit(x_2d, y)

    # generate plot
    plt.close()
    y_preds = lin_reg.predict(x_2d)
    plotting.plot_data(data)
    plt.plot(x, y_preds)    # plot the line
    img_name = plotting.create_plot_img('lin_reg_plot', 'png')

    return img_name, lin_reg
    
def cluster(data, k):
    kmeans = KMeans(n_clusters=k, random_state=42)

    y_preds = kmeans.fit_predict(data)

    # generate plot
    plt.close()
    plotting.plot_decision_boundaries(kmeans, data)
    plotting.plot_data(data)
    plotting.plot_centroids(kmeans.cluster_centers_)
    img_name = plotting.create_plot_img('cluster_plot', 'png')

    return img_name, kmeans

def outlier(data, threshold):
    gm = GaussianMixture(n_components=1, n_init=10, random_state=42)
    gm.fit(data)

    # get outliers
    densities = gm.score_samples(data)
    density_threshold = np.percentile(densities, threshold)
    anomalies = data[densities < density_threshold]

    # generate plot
    plt.close()
    plotting.plot_gaussian_mixture(gm, data)
    plotting.plot_data(data)
    plotting.plot_centroids(gm.means_, gm.weights_)
    plotting.plot_data(anomalies, color='r', marker='*', size=30)
    img_name = plotting.create_plot_img('outlier_plot', 'png')

    return img_name, gm, anomalies