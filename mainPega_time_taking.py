from flask import Flask, request, jsonify
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch
import random

app = Flask(__name__)

# Load pre-trained Pegasus model and tokenizer
# tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-cnn_dailymail')
# model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-cnn_dailymail')
tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large')

# Function to generate summary using Pegasus
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')

# Sample cinema-related dataset (you can replace this with a larger dataset)
cinema_dataset = [
    "film", "movie", "cinema", "actor", "actress", "director", 
    "producer", "screenplay", "script", "scene", "shot", 
    "camera", "dialogue", "plot", "genre", "award", 
    "festival", "box office", "sequel"
]

def augment_input_text(text):
    # return text
    words = text.split()
    augmented_text = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            augmented_text.append(synonym)
        else:
            augmented_text.append(word)
    return ' '.join(augmented_text)

def generate_summary(text, max_words=60, num_candidates=5, temperature=0.7, seed=None):
    if seed:
        torch.manual_seed(seed)
        random.seed(seed)
    
    summaries = []
    for _ in range(num_candidates):
        augmented_text = augment_input_text(text)
        input_ids = tokenizer.encode(augmented_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(input_ids, max_length=512, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, temperature=temperature, do_sample=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    
    selected_summary = random.choice(summaries)
    
    # Post-process the summary to ensure it contains at most 60 words
    summary_words = selected_summary.split()
    if len(summary_words) > max_words:
        selected_summary = ' '.join(summary_words[:max_words])
    
    return selected_summary

@app.route('/summarize', methods=['POST'])
def summarize_text():
    article = request.json['article']
    summary = generate_summary(article, 60)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
