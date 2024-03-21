import torch
import time
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, BartTokenizer, BartForConditionalGeneration
from sklearn.model_selection import train_test_split
from rouge import Rouge
import json

# Load the PEGASUS tokenizer and model
selected_layers=16
device = "cuda" #if torch.cuda.is_available() else "cpu"
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
#model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large").to(device)
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large", num_hidden_layers=selected_layers).to(device)

# Load and preprocess the JSON file
with open("data/airbus_helicopters_train_set.json", "r") as f:
    data = json.load(f)

# Split the data into train 
train_data, test_data = train_test_split(list(data.values()), test_size=0.01, random_state=42)

# Tokenize the text and generate summaries
def tokenize_and_generate_summary(data):
    inputs = tokenizer([d["original_text"] for d in data], return_tensors="pt", truncation=True, padding=True).to(device)
    labels = tokenizer([d["reference_summary"] for d in data], return_tensors="pt", truncation=True, padding=True).to(device)

    return inputs, labels

train_inputs, train_labels = tokenize_and_generate_summary(train_data)
test_inputs, test_labels = tokenize_and_generate_summary(test_data)

# Define training parameters
epochs = 6
batch_size = 4
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
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
    summaries = model.generate(**inputs, max_length=80, num_beams=4, length_penalty=1.0, early_stopping=True)
    return [tokenizer.decode(summary, skip_special_tokens=True) for summary in summaries]

"""generated_summaries = generate_summaries(test_data)

# Calculate ROUGE scores
rouge = Rouge()
reference_summaries = [d["reference_summary"] for d in test_data]
scores = rouge.get_scores(generated_summaries, reference_summaries, avg=True)

print("ROUGE scores:", scores)"""
with open("data/test_set.json", "r") as f:
    testdata = json.load(f)
generated_summaries = {}

# Record the start time
start_time = time.time()

# Generate summaries
for train_id, data_item in testdata.items():
    original_text = data_item["original_text"]
    generated_summary = generate_summaries(original_text)
    
    # Create a unique identifier for the summary
    uid = f"test_{str(train_id[5:])}"
    
    # Add generated summary to the dictionary
    generated_summaries[uid] = {
        "generated_summary": generated_summary[0],
        "uid": uid
    }

with open("generated_test.json", "w", encoding="utf-8") as outfile:
    json.dump(generated_summaries, outfile, indent=4)

end_time = time.time()
print(end_time - start_time)

# Uncomment for model parameters:

#for name, param in model.named_parameters():
#        if param.requires_grad:
#            print(f"Layer: {name}, Size: {param.size()}, Parameters: {param.numel()}")
