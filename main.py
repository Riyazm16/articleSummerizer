from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask import Flask, request, jsonify
import random

# Model and tokenizer initialization (load once at startup)
model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Flask app setup
app = Flask(__name__)

@app.route("/summarize", methods=["POST"])
def summarize():
    # Get article text from request
    article = request.json["article"]

    # Input preparation
    inputs = tokenizer(
        f"summarize: {article}", return_tensors="pt", max_length=512, truncation=True
    )

    # Generate multiple summaries with temperature variations
    summaries = []
    for _ in range(3):  # Generate 3 summaries with different randomness
        temperature = random.uniform(0.7, 1.3)  # Vary temperature for diverse outputs
        output = model.generate(
            inputs["input_ids"],
            max_length=60,
            min_length=30,
            num_beams=2,
            temperature=temperature,
        )
        summaries.append(tokenizer.decode(output[0], skip_special_tokens=True))

    # Select the best summary by length and uniqueness
    best_summary = max(summaries, key=lambda s: len(s))  # Choose longest summary
    if len(set(summaries)) > 1:  # If summaries differ, choose longest
        while best_summary in summaries[:-1]:  # Check for duplicates
            summaries.remove(best_summary)
            best_summary = max(summaries, key=lambda s: len(s))

    return jsonify({"summary": best_summary})

if __name__ == "__main__":
    app.run(debug=True)
