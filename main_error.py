from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config,GPT2Model
import torch

app = Flask(__name__)

embed_dim = 768
num_heads = 16

# Ensure embed_dim is divisible by num_heads
if embed_dim % num_heads != 0:
    num_heads = embed_dim // (embed_dim // num_heads)

config = GPT2Config(embed_dim=embed_dim, num_heads=num_heads)

model = GPT2LMHeadModel(config)

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

model = GPT2Model.from_pretrained("EleutherAI/gpt-neo-2.7B",config=config)
# tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
# model = GPT2LMHeadModel.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Function to generate summary using GPT-2
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
        summary_ids = model.generate(input_ids, max_length=512, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, temperature=temperature)
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
    app.run(debug=True, port=5001)  # Specify the desired port number
