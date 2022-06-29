from re import TEMPLATE
from flask import Flask, render_template, redirect, url_for, request
from flask_cors import cross_origin, CORS
from engine.engine.Model import ChessEngine

app = Flask(__name__)

app.config.update(
    TESTING=True,
    TEMPLATES_AUTO_RELOAD=True,
    DEBUG=True
)

train = False

app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'


cors = CORS(app, origins=["http://localhost:8080", "http://matiraj.me"])

chess_engine = ChessEngine()
if train:
    chess_engine.train(max_rows=1000000, epochs=20)
    chess_engine.save()
else:
    chess_engine.load()




@app.route("/")
def hello_world():
    return render_template('index.html')


@app.route("/model")
def model():
    print("a")
    return 'a'


@app.route("/train")
def train():
    chess_engine.train()
    return render_template('index.html')


@app.route("/evaluate", methods=["POST"])
@cross_origin(headers=['Content- Type', 'Authorization'])
def evaluate():
    if request.method == "POST":
        fen = str(request.data)
        fen = fen[2:len(fen) - 1]
        result = chess_engine.evaluate_fen(fen)
        return str(result[0][0])


@app.route("/img/<path:img_path>")
def pieces(img_path):
    return redirect(url_for('static', filename="/img/" + img_path))
