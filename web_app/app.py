from io import StringIO
import os

from flask import Flask, redirect, render_template, request, url_for, session
from flask_session import Session   # https://flask-session.readthedocs.io/en/latest/

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import plotting
import ml

import openai

app = Flask(__name__)

# Using server side sessions for larger cookie space
app.secret_key = os.getenv('SECRET_KEY')
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

openai.api_key = os.getenv("OPENAI_API_KEY")

# Need this because of the main thread error, switching to backend that does not require GUI
mpl.use('Agg')  

@app.route("/", methods=["GET"])
def index():
    session.clear()
    return render_template("index.html")


@app.route("/config", methods=['POST'])
def config():

    dataset = request.files['dataset']  # get the uploaded dataset from the form
    bytes = dataset.read()
    data = bytes.decode()   # convert bytes to string
    df = pd.read_csv(StringIO(data))    # must make data into a file like obj for Pandas

    session['data'] = data
    metadata = get_data_info(df)

    # include url_for to get the url for the analyze view
    return render_template('config.html', metadata=metadata)


@app.route("/analyze", methods=["POST"])
def analyze():

    session['title'] = request.form['title']
    session['x-axis'] = request.form['x-axis']
    session['y-axis'] = request.form['y-axis']
    
    # Extract the selected columns from the full dataframe
    full_data = pd.read_csv(StringIO(session['data']))
    scatter_data = full_data[[session['x-axis'], session['y-axis']]]

    # Create the plot image
    plt.close()
    plotting.plot_data(scatter_data)
    img_name= plotting.create_plot_img('plot', 'png')
        
    return render_template("analyze.html", img_name=img_name)


def get_data_info(data):
    data_info = {
        'num_cols': len(data.columns),
        'column_names': str(data.columns),
        'columns': data.columns,
    }
    return data_info


@app.route("/generate", methods=["POST"])
def generate():

    full_data = pd.read_csv(StringIO(session['data']))

    # Get all checked inputs from form
    analysis = [key for key in request.form.keys()]

    prompt = 'Generate an engaging caption for a plot titled "{}" with x-axis labeled as "{}" and y-axis labeled as "{}".'.format(
        session['title'], 
        session['x-axis'], 
        session['y-axis'])

    if 'cluster' in analysis:
        type = 'cluster'
        num_clusters = int(request.form['num-clusters'])
        percentage = float(request.form['cluster-percentage'])
        img, model, anomalies = ml.cluster(full_data, percentage)

    elif 'regression' in analysis:
        type = 'regression'
        percentage = float(request.form['regression-percentage'])
        img, model, anomalies = ml.regression(full_data, percentage)

    return render_template('results.html',
        img=img,
        model=model,
        anomalies=anomalies,
        # query=query,
        # response=response
    )


@app.route("/prompt", methods=["POST"])
def prompt():
    print(request)
    

# @app.route('/query', methods=['POST'])
# def query():
#     query = 'Generate an engaging caption for a plot titled "{}" with x-axis labeled as "{}" and y-axis labeled as "{}".'.format(
#         session['title'], 
#         session['x-axis'], 
#         session['y-axis']
#     )

#     response = openai.Completion.create(
#         model='text-davinci-002',
#         prompt=query,
#         temperature=0.5,
#         max_tokens=2048
#     )

#     return query, response.choices[0].text
