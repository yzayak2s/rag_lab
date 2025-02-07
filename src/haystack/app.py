# Importing required libraries
from flask import Flask

app = Flask(__name__)


@app.route('/')
def welcome():
    return "<h1>Welcome to your rag application!</h1>"

if __name__ == '__main__':
    app.run(debug=True)