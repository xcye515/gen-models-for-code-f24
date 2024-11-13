from transformers import RobertaTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import os
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from evaluate import load


### Preprocessing ###

input_path = "rust.txt"
output_file = "rust_processed.txt"

def preprocess_rust_code(input_path, output_filepath, max_length=512):
    with open(output_filepath, "w", encoding="utf-8") as outfile:
        if os.path.isfile(input_path):
            try:
                with open(input_path, "r", encoding="utf-8") as infile:
                    code = infile.read()
                    tokenized = tokenizer(
                        code,
                        max_length=max_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    outfile.write(tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=True) + "\n")
                    print(f"Processed file: {input_path}")
            except Exception as e:
                print(f"Error processing {input_path}: {e}")
        elif os.path.isdir(input_path):
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.endswith(".rs"):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, "r", encoding="utf-8") as infile:
                                code = infile.read()
                                # Tokenize and truncate
                                tokenized = tokenizer(
                                    code,
                                    max_length=max_length,
                                    truncation=True,
                                    padding="max_length",
                                    return_tensors="pt"
                                )
                                outfile.write(tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=True) + "\n")
                                print(f"Processed: {file_path}")
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")
        else:
            print(f"Error: {input_path} is neither a file nor a directory.")

preprocess_rust_code(input_path, output_file)

### Context and Tokenization ###

tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
processed_code_path = "rust_processed.txt"

def preprocess_multiline_code_completion(file_path, num_lines=20):
    inputs = []
    targets = []

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

        for i in range(len(lines) - num_lines):
            input_snippet = " ".join(line.strip() for line in lines[i:i + num_lines - 1]).strip()
            target_snippet = lines[i + num_lines - 1].strip()

            # Skip empty inputs or targets
            if len(input_snippet) < 5 or len(target_snippet) < 5:
                continue

            inputs.append(input_snippet)
            targets.append(target_snippet)

    print(f"Processed {len(inputs)} multiline code snippets.")
    return inputs, targets

inputs, targets = preprocess_multiline_code_completion(processed_code_path, num_lines=3)

for i in range(5):
    print(f"Input {i + 1}:\n{inputs[i]}")
    print(f"Target {i + 1}:\n{targets[i]}")
    print("-" * 50)

def tokenize_inputs_and_targets(inputs, targets, tokenizer, max_length=512):
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        targets,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_data = tokenize_inputs_and_targets(inputs, targets, tokenizer)

dataset = Dataset.from_dict(tokenized_data)

# Split the dataset into training and validation
train_inputs, val_inputs, train_targets, val_targets = train_test_split(
    inputs, targets, test_size=0.2, random_state=42  # 80% training, 20% validation
)

train_data = tokenize_inputs_and_targets(train_inputs, train_targets, tokenizer)
val_data = tokenize_inputs_and_targets(val_inputs, val_targets, tokenizer)

train_dataset = Dataset.from_dict(train_data)
val_dataset = Dataset.from_dict(val_data)

### Padding ###

def preprocess_static_padding(inputs, targets, tokenizer, max_length=512):
    """
    Tokenize inputs and targets with static padding.
    """
    tokenized_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    tokenized_targets = tokenizer(
        targets,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": tokenized_targets["input_ids"],
    }


### Model ###
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-small")


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

save_directory = "./fine-tuned-T5"
trainer.save_model(save_directory)
tokenizer.save_pretrained(save_directory)



### Evaluation ###


def evaluate_model(model, tokenizer, eval_dataset, max_length=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    references = []

    print("Starting evaluation...")
    for i, example in enumerate(tqdm(eval_dataset)):
        # Skip invalid or short inputs
        if len(example["input_ids"]) < 5:
            print(f"Skipping example {i+1} due to short input.")
            continue

        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                output_ids = model.generate(input_ids, max_length=max_length)
        except Exception as e:
            print(f"Error generating prediction for example {i+1}: {e}")
            continue

        predicted_code = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        reference_code = tokenizer.decode(example["labels"], skip_special_tokens=True).strip()

        if not predicted_code or not reference_code:
            print(f"Skipping example {i+1} due to null or empty output.")
            continue

        print(f"Example {i+1}:")
        print(f"Generated Code: {predicted_code}")
        print(f"Reference Code: {reference_code}")
        print("-" * 50)

        predictions.append(predicted_code)
        references.append(reference_code)

    if not predictions:
        print("No valid predictions were generated. Check the model or input dataset.")
        return {}

    bleu_predictions = [pred.split() for pred in predictions]
    bleu_references = [[ref.split()] for ref in references]

    bleu = load("bleu")
    rouge = load("rouge")

    bleu_score = bleu.compute(predictions=bleu_predictions, references=bleu_references)
    rouge_score = rouge.compute(predictions=predictions, references=references)

    return {"BLEU": bleu_score, "ROUGE": rouge_score}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

metrics = evaluate_model(model, tokenizer, val_dataset)
print("Evaluation Metrics:", metrics)

