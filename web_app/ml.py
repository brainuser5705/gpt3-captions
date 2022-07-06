import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import common

from sklearn.linear_model import LinearRegression

# To make consistent results for the same uploaded dataset
np.random.seed(42)

def process_data(data):
    x = data[data.columns[0]]
    y = data[data.columns[1]]
    return x, y

def regression(data):

    x, y = process_data(data)

    # X must be reshaped into a 2d array
    x_2d = x.to_numpy().reshape(x.shape[0], 1)
    lin_reg = LinearRegression()
    lin_reg.fit(x_2d, y)

    # generate plot
    y_preds = lin_reg.predict(x_2d)
    common.plot_data(data)
    plt.plot(x, y_preds)
    common.create_plot_img('lin_reg_plot', 'png')

    return lin_reg
    
def cluster():
    pass

def outlier():
    pass