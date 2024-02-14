from flask import Flask, request, jsonify
import random

app = Flask(__name__)

# Load model directly
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

def generate_summary(article, max_length=150, temperature=0.7):
    temperature = round(random.uniform(0.0, 0.7), 2)  # Vary temperature for diverse outputs
    inputs = tokenizer.encode(article, return_tensors="pt",   max_length=412, truncation=True)
    summary_ids = model.generate(inputs, min_length=90, max_length=max_length, temperature=temperature, length_penalty=5.0, num_beams=4, early_stopping=True,do_sample=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@app.route('/summarize', methods=['POST'])
def summarize_text():
    article = request.json['article']
    summary = generate_summary(article)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
