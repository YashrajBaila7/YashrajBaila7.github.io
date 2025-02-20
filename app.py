from flask import Flask, render_template, request, jsonify, session
import torch
from bpe import Encoder
from GPT2LM import gpt2  # Adjust the import as needed

# Configuration parameters
block_size = 124
vocab_size = 938

# Set paths to your encoder and merges JSON files
E_PATH = "C:\\Users\\DeLL\\OneDrive\\Desktop\\WebApp\\OUT\\encoder.json"
M_PATH = "C:\\Users\\DeLL\\OneDrive\\Desktop\\WebApp\\OUT\\merges.json"
MOD_PATH = "C:\\Users\\DeLL\\OneDrive\\Desktop\\WebApp\\OUT\\model.pt"

# Configure the model
model_config = gpt2.GPT.get_default_config()
model_config.model_type = 'gpt-micro'
model_config.vocab_size = vocab_size
model_config.block_size = block_size
model = gpt2.GPT(model_config)  # Replace with your actual path

gpt2.load(MOD_PATH, model)
max_len = 200  # Maximum generation length

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Instantiate the encoder and load your model
enc = Encoder(E_PATH, M_PATH)
model.to('cuda')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form.get("message", "").strip()
    if not user_input:
        return jsonify({"answer": ""})

    # Initialize conversation history for this user if not present
    if 'conversation_history' not in session:
        session['conversation_history'] = []

    conversation_history = session['conversation_history']

    # Append the new user query with tokens
    conversation_history.append(
        f"<|start_query|> Query: {user_input} <|end_query|>")

    # Build the full prompt from the conversation history plus the answer cue.
    # For example, if the history already contains:
    # <|start_query|> Query: Can students volunteer for social causes? <|end_query|>
    # <|start_answer|> Answer: Volunteer opportunities are available with campus clubs and local NGOs for community service. <|end_answer|>
    # Then the new query will be appended and the prompt becomes:
    # ... [previous conversation]
    # <|start_query|> Query: What food options are available on campus? <|end_query|>
    # <|start_answer|> Answer:
    prompt = "\n".join(conversation_history) + "\n<|start_answer|> Answer: "

    # Encode the prompt using the BPE encoder
    prompt_encoded = torch.tensor(enc.encode(
        prompt), dtype=torch.long, device='cuda').unsqueeze(0)

    # Generate a response using your model
    res = model.generate(prompt_encoded, max_len)

    # Decode the generated tokens into text
    generated_text = enc.decode(res[0].tolist())

    # Extract the answer between the <|start_answer|> and <|end_answer|> tokens.
    answer_start = generated_text.find("<|start_answer|> Answer: ")
    if answer_start != -1:
        answer_start += len("<|start_answer|> Answer: ")
    else:
        answer_start = len(prompt)

    end_token_index = generated_text.find("<|end_answer|>", answer_start)
    if end_token_index == -1:
        # If the end token is missing, fallback by taking text until the first newline
        answer_text = generated_text[answer_start:].split("\n")[0].strip()
    else:
        answer_text = generated_text[answer_start:end_token_index].strip()

    # Append the bot's answer to the conversation history
    conversation_history.append(
        f"<|start_answer|> Answer: {answer_text} <|end_answer|>")

    # Update the session with the new conversation history
    session['conversation_history'] = conversation_history

    return jsonify({"answer": answer_text})


if __name__ == "__main__":
    app.run(debug=True)
