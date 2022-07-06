import os
import matplotlib.pyplot as plt

def plot_data(data, title='', xlabel='', ylabel=''):
    """
    Assumes that data is a Pandas dataframes with two columns
    """
    plt.scatter(data.iloc[:,0], data.iloc[:,1], s=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def create_plot_img(fig_name, fig_extension):
    img_name = fig_name + '.' + fig_extension
    plt.savefig(os.path.join('static', img_name), format=fig_extension, dpi=300)
    return img_name