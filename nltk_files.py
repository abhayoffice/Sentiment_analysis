import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from textblob import TextBlob
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import nltk
nltk.download('wordnet')


def getPolarity(text):
  return TextBlob(text).sentiment.polarity


def analysis(score):
  if score<0:
    return 'Negative'
  elif score==0:
    return 'Neutral'
  else:
    return 'Positive'


# def convert_the_text_file(file):
#     # Read the text file
#     with open(file, 'r') as f:
#         lines = f.readlines()
#
#     # Create a DataFrame with the 'review' column
#     df = pd.DataFrame({'review': lines})
#
#     # Save the DataFrame to a CSV file
#     # data = df.to_csv('output.csv', index=False)
#
#     data = pd.read_csv(df.to_csv(index=False))
#
#     # Split the paragraphs into sentences
#     data['review'] = data['review'].str.split(r'[.!?]')
#
#     # Explode the sentences into separate rows
#     data = data.explode('review')
#
#     # Reset the index
#     data = data.reset_index(drop=True)
#
#     # Convert the 'review' column to strings
#     data['review'] = data['review'].astype(str)
#
#     data['Polarity'] = data['review'].apply(getPolarity)
#
#     data['Analysis'] = data['Polarity'].apply(analysis)
#
#     return data


def convert_the_text_file(file):
    try:
        # Read the text file
        lines = [line.decode('utf-8') for line in file.readlines()]  # Decode bytes to strings

        # Create a DataFrame with the 'review' column
        df = pd.DataFrame({'review': lines})

        # Split the paragraphs into sentences
        df['review'] = df['review'].str.split(r'[.!?]')

        # Explode the sentences into separate rows
        df = df.explode('review')

        # Reset the index
        df = df.reset_index(drop=True)

        # Convert the 'review' column to strings
        df['review'] = df['review'].astype(str)

        # Calculate polarity and sentiment analysis
        df['Polarity'] = df['review'].apply(getPolarity)
        df['Analysis'] = df['Polarity'].apply(analysis)

        return df

    except Exception as e:
        # If any error occurs during file processing, print the error and return None
        print(f"Error occurred: {e}")
        return None

def get_positive_negative_neutral_percentage(data):


    condition = data['Polarity'] > 0
    positive_reviews = data[condition]

    condition = data['Polarity'] < 0
    negative_reviews = data[condition]

    condition_neutral = data['Polarity'] == 0
    neutral_reviews = data[condition_neutral]

    total_reviews = len(data)
    positive_percentage = len(positive_reviews) / total_reviews * 100
    negative_percentage = len(negative_reviews) / total_reviews * 100
    neutral_percentage = len(neutral_reviews) / total_reviews * 100


    # Assuming positive_percentage and negative_percentage are float values
    positive_percentage_rounded = round(positive_percentage)
    negative_percentage_rounded = round(negative_percentage)
    neutral_percentage_rounded = round(neutral_percentage)


    return {"positive":positive_percentage_rounded,
            "negative":negative_percentage_rounded,
            "neutral": neutral_percentage_rounded}


# def get_positive_negative_words(data):
#     lemmatizer = WordNetLemmatizer()
#     nltk.download('stopwords')
#     all_stopwords = set(stopwords.words('english'))
#     all_stopwords.remove('not')
#
#     corpus_positive = []
#     corpus_negative = []
#
#     for review in data['review']:
#         review = re.sub(r'[^A-Za-z]', ' ', review)
#         review = review.lower()
#         review = review.split()
#         review = [lemmatizer.lemmatize(word) for word in review if word not in all_stopwords]
#
#         for word in review:
#             if TextBlob(word).sentiment.polarity > 0:
#                 corpus_positive.append(word)
#             elif TextBlob(word).sentiment.polarity < 0:
#                 corpus_negative.append(word)
#
#     positive_word_freq = nltk.FreqDist(corpus_positive)
#     negative_word_freq = nltk.FreqDist(corpus_negative)
#
#     return {
#         "positive_words": positive_word_freq.most_common(10),
#         "negative_words": negative_word_freq.most_common(10)
#     }

def get_positive_negative_words(data):
    lemmatizer = WordNetLemmatizer()
    nltk.download('stopwords')
    all_stopwords = set(stopwords.words('english'))
    all_stopwords.remove('not')

    corpus_positive = []
    corpus_negative = []

    for review in data['review']:
        review = re.sub(r'[^A-Za-z]', ' ', review)
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if word not in all_stopwords]

        for word in review:
            if TextBlob(word).sentiment.polarity > 0:
                corpus_positive.append(word)
            elif TextBlob(word).sentiment.polarity < 0:
                corpus_negative.append(word)

    positive_word_freq = nltk.FreqDist(corpus_positive)
    negative_word_freq = nltk.FreqDist(corpus_negative)

    # Get all positive and negative words with their occurrences
    positive_words = [f"{count} occurrences of word {word}" for word, count in positive_word_freq.items()]
    negative_words = [f"{count} occurrences of word {word}" for word, count in negative_word_freq.items()]

    # Sort the words based on occurrences
    positive_words.sort(reverse=True)
    negative_words.sort(reverse=True)

    # If needed, you can limit the number of words shown
    # positive_words = positive_words[:10]
    # negative_words = negative_words[:10]

    return {
        "positive_words": positive_words,
        "negative_words": negative_words
    }

def get_tone_of_conversation(positive_percentage, negative_percentage, neutral_percentage):
    max_percentage = max(positive_percentage, negative_percentage, neutral_percentage)

    if max_percentage == positive_percentage:
        return "Positive"
    elif max_percentage == negative_percentage:
        return "Negative"
    else:
        return "Neutral"
