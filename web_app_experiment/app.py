from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'

@app.route("/", methods=['GET'])
def index():

    return render_template("index.html")

@app.route("/upload", methods=['POST'])
def upload():
    
    return redirect(url_for('index'))