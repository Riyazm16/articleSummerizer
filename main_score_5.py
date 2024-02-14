# Import necessary libraries
from flask import Flask, request, jsonify
from summarizer import Summarizer
import random
from logger import get_logger
logger = get_logger()

app = Flask(__name__)

# Initialize the summarizer
model = Summarizer()

@app.route('/summarize', methods=['POST'])
def summarize_text():
    # Get the article from the POST request
    article = request.json['article']

    # Generate the summary
    summary = model(article)

    sentences = summary.split('. ')
    # Shuffle the sentences
    random.shuffle(sentences)
    # Select sentences until the word limit is reached
    summary = ''
    word_count = 0
    for sentence in sentences:
        words = sentence.split()
        if word_count + len(words) > 60:
            break
        summary += sentence + '. '
        word_count += len(words)

    words = summary.split()
    # Limit the summary to the first 60 words
    summary = ' '.join(words[:60])
    logger.info(article)
    logger.info(summary)
    # Return the summary as a JSON response
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
