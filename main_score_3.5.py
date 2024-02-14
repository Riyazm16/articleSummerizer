from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import random
import string
from logger import get_logger
logger = get_logger()
app = Flask(__name__)

def get_article(url):
    # Function to scrape article text from a given URL
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    article_text = ' '.join([p.text for p in soup.find_all('p')])
    return article_text

def preprocess_text(text):
    # Function to preprocess text (remove stopwords, punctuation, etc.)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into text
    cleaned_text = ' '.join(filtered_tokens)
    return cleaned_text

def summarize(text, max_words=60):
    # Function to generate a summary of the given text with a maximum word limit
    sentences = sent_tokenize(text)
    word_frequencies = FreqDist(word_tokenize(preprocess_text(text)))
    ranked_sentences = sorted(sentences, key=lambda s: sum([word_frequencies[w] for w in word_tokenize(s.lower())]), reverse=True)
    
    # Randomly shuffle the sentences
    random.shuffle(ranked_sentences)
    
    summary = ''
    word_count = 0
    for sentence in ranked_sentences:
        words = word_tokenize(sentence)
        if word_count + len(words) <= max_words:
            summary += sentence + ' '
            word_count += len(words)
        else:
            break
    
    return summary.strip()

@app.route('/summarize', methods=['POST'])
def generate_summary():
    data = request.json
    article = data['article']
    summary = summarize(article)
    logger.info(article)
    logger.info(summary)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
