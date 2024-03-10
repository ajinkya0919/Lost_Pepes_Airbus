import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, BartTokenizer, BartForConditionalGeneration
from sklearn.model_selection import train_test_split
from rouge import Rouge
import json

# Load the PEGASUS tokenizer and model
selected_layers=16
device = "cuda" #if torch.cuda.is_available() else "cpu"
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large", num_hidden_layers=selected_layers).to(device)

# Load and preprocess the JSON file
with open("~/airbus_helicopters_train_set.json", "r") as f:
    data = json.load(f)

# Data to train 
train_data, test_data = train_test_split(list(data.values()), test_size=0.01, random_state=42)

# Tokenize the text and generate summaries
def tokenize_and_generate_summary(data):
    inputs = tokenizer([d["original_text"] for d in data], return_tensors="pt", truncation=True, padding=True).to(device)
    labels = tokenizer([d["reference_summary"] for d in data], return_tensors="pt", truncation=True, padding=True).to(device)

    return inputs, labels

train_inputs, train_labels = tokenize_and_generate_summary(train_data)
test_inputs, test_labels = tokenize_and_generate_summary(test_data)

# Setting training parameters
epochs = 6
batch_size = 4
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Model training loop
for epoch in range(epochs):
    for i in range(0, len(train_inputs["input_ids"]), batch_size):
        optimizer.zero_grad()
        input_batch = {k: v[i:i+batch_size] for k, v in train_inputs.items()}
        label_batch = {k: v[i:i+batch_size] for k, v in train_labels.items()}
        outputs = model(input_ids=input_batch["input_ids"], 
                        attention_mask=input_batch["attention_mask"], 
                        labels=label_batch["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Epoch {epoch + 1}, Step {i}: Loss = {loss.item()}")

# Generate summaries for test data
def generate_summaries(data):
    inputs = tokenizer(data, return_tensors="pt", truncation=True, padding=True).to(device)
    summaries = model.generate(**inputs, max_length=50, num_beams=4, length_penalty=1.0, early_stopping=True)
    return [tokenizer.decode(summary, skip_special_tokens=True) for summary in summaries]


generated_summaries = {}

# Generate summaries
for train_id, data_item in data.items():
    original_text = data_item["original_text"]
    generated_summary = generate_summaries(original_text)
    
    # Create a unique identifier for the summary
    uid = f"test{str(train_id[5:])}"
    
    # Add generated summary to the dictionary
    generated_summaries[uid] = {
        "generated_summary": generated_summary[0],
        "uid": uid
    }

# Create a file to store the generated summaries
with open("generated_summaries-pegasus-large-airbus.json", "w", encoding="utf-8") as outfile:
    json.dump(generated_summaries, outfile, indent=4)

