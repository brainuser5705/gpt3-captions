from io import StringIO
import os

from flask import Flask, redirect, render_template, request, url_for, session
from flask_session import Session   # https://flask-session.readthedocs.io/en/latest/

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import common
import ml

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')    # to use for sessions

# Using server side sessions for larger cookie space
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

mpl.use('Agg')  # Need this because of the main thread error, switching to backend that does not require GUI

@app.route("/", methods=("GET", "POST"))
def index():
    session.clear()
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

        session['data'] = new_data.to_csv() 

        plt.close()
        common.plot_data(new_data, title=title, xlabel=xaxis_col, ylabel=yaxis_col)
        img_name= common.create_plot_img('plot', 'png')
            
        return render_template("analyze.html", img_name=img_name)


@app.route("/generate", methods=["POST"])
def generate():

    data = pd.read_csv(StringIO(session['data']))

    if request.form['regression']:
        lin_reg_img, lin_reg = ml.regression(data)

    return render_template('generate.html', lin_reg_img=lin_reg_img)


# Analysis can be done here
def get_data_info(data):
    data_info = {
        'num_cols': len(data.columns),
        'column_names': str(data.columns),
        'columns': data.columns,
    }
    return data_info

