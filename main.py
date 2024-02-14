from flask import Flask, request, jsonify
import random

app = Flask(__name__)

# Load model directly
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForCausalLM,T5Tokenizer, T5ForConditionalGeneration

artickle_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
artickle_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
def generate_summary(article, max_length=150,min_length=90):
    temperature = round(random.uniform(0.0, 0.7), 2)  # Vary temperature for diverse outputs
    inputs = artickle_tokenizer.encode(article, return_tensors="pt",   max_length=412, truncation=True)
    summary_ids = artickle_model.generate(inputs, min_length=min_length, max_length=max_length, temperature=temperature, length_penalty=5.0, num_beams=4, early_stopping=True,do_sample=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@app.route('/summarize', methods=['POST'])
def summarize_text():
    article = request.json['article']
    summary = generate_summary(article,150,90)
    return jsonify({'summary': summary})

# Load model directly

checkpoint="unikei/t5-base-split-and-rephrase"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint)

def headline_rephraser(prompt,max_length=12,min_length=7):
    complex_tokenized = tokenizer(prompt,padding="max_length",truncation=True,                               max_length=256,return_tensors='pt')
    simple_tokenized = model.generate(complex_tokenized['input_ids'], attention_mask = complex_tokenized['attention_mask'], max_length=256, num_beams=5)
    generated_text = tokenizer.batch_decode(simple_tokenized, skip_special_tokens=True)
    return generated_text

@app.route("/rephrase",methods=['POST'])
def rephrase_headline():
    headline = request.json['headline']
    summary = generate_summary(headline,50,120)
    return jsonify({'headline': summary})

if __name__ == '__main__':
    app.run(debug=True)
