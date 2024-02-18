from flask import Flask, request, jsonify
import random
import torch
import random

app = Flask(__name__)

# Load model directly
# Load model directly
from transformers import pipeline

checkpoint="EleutherAI/gpt-neo-125m"
generator = pipeline("text-generation", model=checkpoint, device=0 if torch.cuda.is_available() else -1)

def headline_rephraser(headline):
    temperature = round(random.uniform(0.0, 0.7), 2)  # Vary temperature for diverse outputs
    print(temperature)
    # paraphrased_headline = generator(headline, max_length=40,temperature=temperature,length_penalty=2.0,num_beams=10)[0]['generated_text'].strip()
    paraphrased_headline = generator(headline, max_length=40,num_beams=4,length_penalty=2.0,temperature=temperature)[0]['generated_text'].strip()
    return paraphrased_headline

@app.route("/rephrase",methods=['POST'])
def rephrase_headline():
    headline = request.json['headline']
    summary = headline_rephraser(headline)
    return jsonify({'headline': summary})

if __name__ == '__main__':
    app.run(debug=True)
