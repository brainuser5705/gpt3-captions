from flask import Flask, request, render_template


app = Flask(__name__)

@app.route("/", methods=['GET'])
def index():

    return render_template("index.html")

@app.route("/page", methods=['POST'])
def another_page():
    
    return render_template("another-page.html")