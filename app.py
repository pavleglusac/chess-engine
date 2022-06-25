from re import TEMPLATE
from flask import Flask, render_template, redirect, url_for
from engine import Model

app = Flask(__name__)


app.config.update(
    TESTING=True,
    TEMPLATES_AUTO_RELOAD=True,
    DEBUG=True,
)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/model")
def model():
    a = Model.main()
    print(a)
    return a

@app.route("/img/<path:img_path>")
def pieces(img_path):
    print(img_path)
    return redirect(url_for('static', filename="/img/" + img_path))
