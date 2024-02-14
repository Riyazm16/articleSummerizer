from flask import Flask, request, jsonify
from logger import get_logger

# Get the logger from the logger module
logger = get_logger()
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import torch

app = Flask(__name__)

# Load pre-trained GPT-2 model and tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
# model = GPT2LMHeadModel.from_pretrained("EleutherAI/gpt-neo-125M")
# Function to generate summary using GPT-2
model_name = "Falconsai/text_summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')

import random

# Sample cinema-related dataset (you can replace this with a larger dataset)
cinema_dataset = [
    "film", "movie", "cinema", "actor", "actress", "director", 
    "producer", "screenplay", "script", "scene", "shot", 
    "camera", "dialogue", "plot", "genre", "award", 
    "festival", "box office", "sequel"
]

def augment_input_text(text):
    words = text.split()
    augmented_text = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            augmented_text.append(synonym)
        elif word.lower() in cinema_dataset:
            augmented_text.append(word)
        else:
            augmented_text.append(word)
    return ' '.join(augmented_text)

def generate_summary(text, max_words=60, num_candidates=5, temperature=0.7, seed=None):
    if seed:
        torch.manual_seed(seed)
        random.seed(seed)
    
    augmented_texts = [text] + [augment_input_text(text) for _ in range(num_candidates-1)]
    summaries = []
    for augmented_text in augmented_texts:
        input_ids = tokenizer.encode(augmented_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(input_ids, max_length=512, num_return_sequences=1, pad_token_id=model.config.eos_token_id, temperature=temperature)
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
    logger.info(article)
    summary = generate_summary(article,60)
    logger.info(summary)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
