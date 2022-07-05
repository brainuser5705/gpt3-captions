from io import StringIO
import os

import openai
from flask import Flask, redirect, render_template, request, url_for, session

import matplotlib.pyplot as plt
import pandas as pd


app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route("/", methods=("GET", "POST"))
def index():
    return render_template("index.html")


@app.route("/config", methods=['POST'])
def config():

    if request.method == 'POST':

        dataset = request.files['dataset']  # get the uploaded dataset from the form
        bytes = dataset.read()
        data = bytes.decode()   # convert bytes to string
        df = pd.read_csv(StringIO(data))    # must make data into a file like obj for Pandas

        session['data'] = data

        metadata = get_data_info(df)

        # include url_for to get the url for the analyze view
        return render_template('config.html', metadata=metadata)


@app.route("/analyze", methods=("GET", "POST"))
def analyze():

    if request.method == 'POST':
        
        title = request.form['title']
        xaxis_col = request.form['x-axis']
        yaxis_col = request.form['y-axis']
        
        orig_data = pd.read_csv(StringIO(session['data']))
        new_data = orig_data[[xaxis_col, yaxis_col]] # set 0 to x axis, 1 to y axis

        plot_data(new_data, title=title, xlabel=xaxis_col, ylabel=yaxis_col)
        img_name= create_plot_img('plot', 'png')
            
        return render_template("analyze.html", img_name=img_name)


@app.route("/generate", methods=("GET", "POST"))
def generate():
    pass


# Analysis can be done here
def get_data_info(data):
    data_info = {
        'num_cols': len(data.columns),
        'column_names': str(data.columns),
        'columns': data.columns,
    }
    return data_info

def plot_data(data, title, xlabel, ylabel):
    """
    Assumes that data is a Pandas dataframes with two columns
    """
    plt.scatter(data.iloc[:,0], data.iloc[:,1])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def create_plot_img(fig_name, fig_extension):
    img_name = fig_name + '.' + fig_extension
    plt.savefig(os.path.join('static', img_name), format=fig_extension, dpi=300)
    return img_name