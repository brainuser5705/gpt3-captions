import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Libraries for Linear Regression
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from matplotlib.pyplot import cm

FIG_EXT = 'png'
DPI = 300
LIN_REG_TEMPLATE = open('lin_reg_template.txt', 'r').read()
CLUSTER_TEMPLATE = open('cluster_template.txt', 'r').read()

def generate_filename(x_col, y_col):
    x_name =  ('_'.join(x_col.split())).lower()
    y_name =  ('_'.join(y_col.split())).lower()
    return x_name + '_vs_' + y_name


def save_plot(name, analysis_type):
    plt.savefig(os.path.join('images/', analysis_type, name + '.png'), format=FIG_EXT, dpi=DPI, bbox_inches='tight')


def perform_lin_reg(dataset_name, x_col, y_col, i_col=None, outlier=False):
    
    print('Performing Linear Regression on: ' + dataset_name)

    # Read data
    print('Read data....')
    data = pd.read_csv(dataset_name)
    X = data[x_col]
    Y = data[y_col]

    # Create model
    print('Created model.....')
    X_const = sm.add_constant(X)
    lin_reg  = sm.OLS(Y, X_const).fit()

    # Get outliers
    print('Getting outliers....')
    if outlier:
        data['Stu_residual'] = lin_reg.get_influence().resid_studentized_internal
        outliers = data[abs(data['Stu_residual']) > 3]

    # Get correlation
    print('Getting correlation.....')
    corr, _ = pearsonr(X, Y)

    # Create plot and save
    print('Creating plot......')
    plt.close()
    plt.figure(figsize=(8,8))

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(x_col + ' VS ' + y_col)

    plt.scatter(data[x_col], data[y_col], s=4)
    plt.scatter(outliers[x_col], outliers[y_col], marker='x', c='r')
    preds = lin_reg.predict(X_const)
    plt.plot(X, preds, 'g-')

    plt.tight_layout()

    filename = generate_filename(x_col, y_col)
    save_plot(filename, 'regression')

    print('Writing data.......')
    content = LIN_REG_TEMPLATE.format(
        title=x_col + ' VS ' + y_col,
        dataset_name=dataset_name,
        columns=', '.join(list(data.columns)),
        x_col=x_col,
        y_col=y_col,
        x_min=X.min(), x_max=X.max(),
        y_min=Y.min(), y_max=Y.max(),
        correlation=corr,
        intercept=lin_reg.params[0],
        slope=lin_reg.params[1],
    )

    with open('results/' + filename + '.txt', 'w') as file:
        file.write(content)
        for i in range(len(outliers)):
            out = outliers.iloc[i,:]
            file.write('- ' + str(out[i_col]) + ' (' + str(out[x_col]) + ', ' + str(out[y_col]) + ')\n')

    print('Done')

def perform_cluster(dataset_name, x_col, y_col, i_col=None, eps=0.5, min_samples=5):

    print('Performing DBSCAN on: ' + dataset_name)

    # Read data
    print('Read data....')
    data = pd.read_csv(dataset_name)
    X = data[x_col]
    Y = data[y_col]

    # Scale data
    print('Scaling data.....')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.to_numpy().reshape(-1,1))
    Y_scaled = scaler.fit_transform(Y.to_numpy().reshape(-1,1))

    # Create model
    print('Created model.....')
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(np.c_[X_scaled,Y_scaled])

    # Get outliers
    print('Getting outliers.....')
    outlier_mask = dbscan.labels_ == -1
    outliers = data.iloc[outlier_mask,:]

    # Create plot and save
    print('Creating plot.....')
     # Get all the anomalies instances
    outliers_mask = dbscan.labels_ == -1
    outliers = data[outliers_mask]
    
    # Get clusters
    clusters = np.unique(dbscan.labels_)[1:]

    # Rank cluster by size
    sizes = {}
    for cluster in clusters:
        sizes[cluster] = len(dbscan.labels_[dbscan.labels_==cluster])
    
    # Plot each point, labeled and colored by its cluster
    color = cm.Set2(np.linspace(0,1,len(clusters)))
    for clust, c in zip(clusters, color):
        points = data[dbscan.labels_==clust]
        plt.scatter(points[x_col], points[y_col], color=c, s=20, label=clust)
    
    plt.scatter(outliers[x_col], outliers[y_col], c='r', s=30, marker='x')

    plt.tight_layout()
    
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(x_col + ' VS ' + y_col)
    
    # places legend to right of plot, taken from stackoverflow
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Cluster')
    
    filename = generate_filename(x_col, y_col)
    save_plot(filename, 'cluster')

    # write data
    print('Writing data.......')
    content = CLUSTER_TEMPLATE.format(
        title=x_col + ' VS ' + y_col,
        dataset_name=dataset_name,
        columns=', '.join(list(data.columns)),
        x_col=x_col,
        y_col=y_col,
        x_min=X.min(), x_max=X.max(),
        y_min=Y.min(), y_max=Y.max(),
        num_clusters=len(dbscan.labels_[dbscan.labels_==-1]),
        eps=eps,
        min_samples=min_samples,
    )

    with open('results/cluster/' + filename + '.txt', 'w') as file:
        file.write(content)
        for i, size in enumerate(sorted(sizes.items(), reverse=True, key=lambda item: item[1])):
            file.write(str(i) + '. ' + str(size) + '\n')
        file.write('\nOutliers:\n')
        for i in range(len(outliers)):
            out = outliers.iloc[i,:]
            file.write('- ' + str(out[i_col]) + ' (' + str(out[x_col]) + ', ' + str(out[y_col]) + ')\n')

    print('Done')