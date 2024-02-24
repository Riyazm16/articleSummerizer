from flask import Flask, request, jsonify
import random
app = Flask(__name__)
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,PegasusForConditionalGeneration, PegasusTokenizer
import torch


tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
rephrase_tokenizer = PegasusTokenizer.from_pretrained(model_name)
rephrase_model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)


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

def headline_rephraser(headline,num_return_sequences):
  batch = rephrase_tokenizer.prepare_seq2seq_batch([headline],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  temperature = round(random.uniform(0.0, 1.5), 2)  # Vary temperature for diverse outputs
  translated = rephrase_model.generate(**batch,min_length=12,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=temperature)
  tgt_text = rephrase_tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

@app.route("/rephrase",methods=['POST'])
def rephrase_headline():
    headline = request.json['headline']
    summary = headline_rephraser(headline,10)
    return jsonify({'headline': summary})


if __name__ == '__main__':
    app.run(debug=True)
