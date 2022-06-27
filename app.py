from io import StringIO
import os

import openai
from flask import Flask, redirect, render_template, request, url_for

import pandas as pd

import ast

ALLOWED_DATASET_EXTENSIONS = {'csv'}
ALLOWED_VISUALIZATION_EXTENSIONS = {'jpg', 'png'}

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":

        # get the uploaded dataset from the form
        dataset = request.files['dataset']
        # Read in the bytes and decode to string
        content = dataset.read().decode()
        # Convert to a file-like object for Pandas
        content = StringIO(content)

        # animal = request.form["animal"]
        # response = openai.Completion.create(
        #     model="text-davinci-002",
        #     prompt=generate_prompt(animal),
        #     temperature=0.6,
        # )
        #return redirect(url_for("index", result=response.choices[0].text))

        return redirect(url_for("index", result=get_data_info(content)))

    result = request.args.get("result")
    if result is None:  # At startup
        result = {}
    else:
        # convert the string into a dict
        result = ast.literal_eval(result)   
    return render_template("index.html", result=result)

# Analysis can be done here
def get_data_info(file_obj):
    data = pd.read_csv(file_obj)
    data_info = {
        'num_cols': len(data.columns),
        'column_names': str(data.columns),
    }
    return data_info

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
