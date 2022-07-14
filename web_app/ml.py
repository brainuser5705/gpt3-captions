import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import plotting

from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture

from flask import session

# To make consistent results for the same uploaded dataset
np.random.seed(42)

def process_data(data):
    x = data[session['x-axis']]
    y = data[session['y-axis']]
    return x, y

def regression(data, percentage=0):

    x, y = process_data(data)

    # X must be reshaped into a 2d array
    x_2d = x.to_numpy().reshape(x.shape[0], 1)
    lin_reg = LinearRegression()
    lin_reg.fit(x_2d, y)

    y_preds = lin_reg.predict(x_2d)

    # reshape arrays to 2d array with each value being its own array
    x = x.to_numpy().reshape(-1,1)
    y_preds = y_preds.reshape(-1,1)
    preds = np.concatenate((x, y_preds), axis=1)
    actual = data[[session['x-axis'], session['y-axis']]].to_numpy()
    dists = abs((actual-preds).reshape(1,-1)[0][1::2])    # distances are now in a 1d array

    # obtain outliers
    data = data.assign(distance=dists).sort_values(by='distance', ascending=False)
    data = data.drop('distance', axis=1)    # so it does not interfere with plotting
    num_anomalies = int(len(data) * (percentage/100))
    anomalies = data.iloc[:num_anomalies]

    # generate plot
    plt.close()
    plotting.plot_data(data)
    plt.plot(x, y_preds)    # plot the line
    plotting.plot_data(anomalies, color='r', marker='*', size=30) # plot anomalies
    img_name = plotting.create_plot_img('lin_reg_plot', 'png')
    return img_name, lin_reg, anomalies
    

def cluster(data, percentage=0):
    gm = GaussianMixture(n_components=1, n_init=10, random_state=42)
    gm.fit(data)

    # get outliers
    densities = gm.score_samples(data)
    density_threshold = np.percentile(densities, percentage)
    anomalies = data[densities < density_threshold]

    # generate plot
    plt.close()
    plotting.plot_gaussian_mixture(gm, data)
    plotting.plot_data(data)
    plotting.plot_centroids(gm.means_, gm.weights_)
    plotting.plot_data(anomalies, color='r', marker='*', size=30)
    img_name = plotting.create_plot_img('cluster_plot', 'png')

    return img_name, gm, anomalies