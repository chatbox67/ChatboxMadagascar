from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

app = Flask(__name__)

# Modèle léger local
model_name = "gpt2"  # ou "databricks/dolly-v2-3b" quantifié si possible
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Charger l'historique des conversations
try:
    with open("conversations.json", "r") as f:
        conversations = json.load(f)
except:
    conversations = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    
    # Ajouter au contexte
    conversations.append({"role": "user", "content": user_input})
    
    # Générer réponse
    input_text = "\n".join([f"{c['role']}: {c['content']}" for c in conversations])
    inputs = tokenizer.encode(input_text + "\nAI:", return_tensors="pt")
    outputs = model.generate(inputs, max_length=inputs.shape[1]+100, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split("AI:")[-1].strip()
    
    # Sauvegarder la réponse
    conversations.append({"role": "AI", "content": answer})
    with open("conversations.json", "w") as f:
        json.dump(conversations, f)
    
    return jsonify({"reply": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)