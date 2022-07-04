from io import StringIO
import os

import openai
from flask import Flask, redirect, render_template, request, url_for

import matplotlib.pyplot as plt

import pandas as pd

from flask import make_response

ALLOWED_DATASET_EXTENSIONS = {'csv'}
ALLOWED_VISUALIZATION_EXTENSIONS = {'jpg', 'png'}

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":

        # get the uploaded dataset from the form
        dataset = request.files['dataset']
        # Read in the bytes
        bytes = dataset.read()
        print('Bytes', bytes)

        plot_data(bytes_to_df(bytes))
        img_name= create_plot_img('plot', 'png')

        # include url_for to get the url for the analyze view
        resp = make_response(redirect(url_for('analyze')))
        resp.set_cookie('dataset', dataset.read())
        resp.set_cookie('img_name', img_name)

        return resp

    return render_template("index.html")

@app.route("/analyze", methods=("GET", "POST"))
def analyze():

    img_name = request.cookies.get('img_name')

    if request.method == 'POST':

        # convert the dataset into df then into bytes
        # upload the dataset?
        print('Data:', request.cookies.get('dataset'))
        print(type(request.cookies.get('dataset')))

        data = bytes_to_df(request.cookies.get('dataset'))
        # access the inputs through request.form which is a dictionary of the name-value pairs

        metadata = get_data_info(data)
        return render_template('analyze.html', img_name=img_name,metadata=metadata)
        
    return render_template("analyze.html", img_name = img_name)


@app.route("/generate", methods=("GET", "POST"))
def generate():
    pass


def bytes_to_df(bytes):
    # Decode bytes into string
    decode = bytes.decode()
    # convert to 'file-like obj' for Pandas's read_csv
    return pd.read_csv(StringIO(decode))

# Analysis can be done here
def get_data_info(data):
    data_info = {
        'num_cols': len(data.columns),
        'column_names': str(data.columns),
    }
    return data_info

def plot_data(data):
    """
    Assumes that data is a Pandas dataframes with two columns
    """
    plt.scatter(data.iloc[:,0], data.iloc[:,1])

def create_plot_img(fig_name, fig_extension):
    img_name = fig_name + '.' + fig_extension
    plt.savefig(os.path.join('static', img_name), format=fig_extension, dpi=300)
    return img_name


def generate_prompt(animal):
    return """Suggest three names for an animal that is a superhero.

Animal: Cat
Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
Animal: Dog
Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
Animal: {}
Names:""".format(
        animal.capitalize()
    )
