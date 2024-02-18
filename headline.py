import torch
import random
from transformers import pipeline

def generate_paraphrased_headline(prompt):
    temperature = round(random.uniform(0.0, 1.5), 2)  # Vary temperature for diverse outputs
    generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B", device=0 if torch.cuda.is_available() else -1)
    paraphrased_headline = generator(prompt, max_length=50, min_length=10,temperature=temperature,length_penalty=5.0,num_beams=10)[0]['generated_text'].strip()
    return paraphrased_headline

# Example usage
prompt = "Dunki on Netflix - Shah Rukh Khan humorously claims to have taught South Koreans about love; shows a finger heart to BTS"
paraphrased_headline = generate_paraphrased_headline(prompt)

print("Original Headline:", prompt)
print("Paraphrased Headline:", paraphrased_headline)
