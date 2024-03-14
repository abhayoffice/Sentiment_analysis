import os
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from nltk_files import convert_the_text_file, get_positive_and_negative_percentage, get_positive_negative_words

app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = '/path/to/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def test():
    return jsonify({"status":"OK TESTED by Pandey ji!!"})

@app.route("/evaluate_sentiment", methods=['POST'])
def evaluate_sentiment_analysis():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    # Check if the file extension is allowed
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Save the uploaded file to the upload folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Convert text file to DataFrame
        data = convert_the_text_file(file_path)

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
    else:
        return jsonify({"error": "Invalid file extension"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8090, debug=True)
