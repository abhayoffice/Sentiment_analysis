import os
from flask import Flask, jsonify
from functions_file import convert_the_text_file, get_positive_and_negative_percentage, get_positive_negative_words

app = Flask(__name__)

@app.route("/")
def test():
    return jsonify({"status":"OK TESTED by Pandey ji!!"})

@app.route("/evaluate_sentiment")
def evaluate_sentiment_analysis():
    # Convert text file to DataFrame
    data = convert_the_text_file("/content/transcribed_file.txt")

    # Get positive/negative percentages
    percentages = get_positive_and_negative_percentage(data)

    # Get positive/negative words
    words = get_positive_negative_words(data)

    return jsonify({
        "positive_percentage": percentages["positive"],
        "negative_percentage": percentages["negative"],
        "positive_words": words["positive_words"],
        "negative_words": words["negative_words"]
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8090, debug=True)
