import os
import json
from flask import Flask, request, Response, session, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)

@app.route("/")
def test():
    return jsonify({"status":"OK TESTED by Pandey ji!!"})

@app.route("/evaluate_sentiment")
def evaluate_sentiment_analysis():


    pass

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8090, debug=True)
