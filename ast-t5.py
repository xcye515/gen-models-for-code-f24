import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer

from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import tree_sitter
from tree_sitter import Language, Parser

Language.build_library(
    'build/my-languages.so',
    ['tree-sitter-rust']
)
RUST_LANGUAGE = Language('build/my-languages.so', 'rust')

parser = Parser()
parser.set_language(RUST_LANGUAGE)


### AST Construction ###
def code_to_ast(code_snippet):
    tree = parser.parse(bytes(code_snippet, "utf-8"))
    root_node = tree.root_node

    def traverse(node):
        if node.is_named:
            children = [traverse(child) for child in node.children]
            return f"({node.type} {' '.join(children)})"
        else:
            return node.type

    return traverse(root_node)

preprocessed_code_path = "rust_processed.txt"

def preprocess_ast_dataset(file_path, num_lines=20, tokenizer=None):
    inputs = []
    targets = []
    
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        
        for i in range(len(lines) - num_lines):
            input_snippet = "\n".join(line.strip() for line in lines[i:i + num_lines - 1])
            target_snippet = lines[i + num_lines - 1].strip()
            
            # Convert the input snippet to its AST representation
            input_ast = code_to_ast(input_snippet)
            if input_ast and len(input_ast) > 5 and len(target_snippet) > 5:
                inputs.append(input_ast)
                targets.append(target_snippet)
    
    tokenized_data = {
        "input_ids": tokenizer(inputs, max_length=512, truncation=True, padding="max_length")["input_ids"],
        "attention_mask": tokenizer(inputs, max_length=512, truncation=True, padding="max_length")["attention_mask"],
        "labels": tokenizer(targets, max_length=512, truncation=True, padding="max_length")["input_ids"],
    }
    
    return tokenized_data



### Model ###
tokenizer = AutoTokenizer.from_pretrained("gonglinyuan/ast_t5_base", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained("gonglinyuan/ast_t5_base", trust_remote_code=True)

data = preprocess_ast_dataset(preprocessed_code_path, num_lines=3, tokenizer=tokenizer)

train_inputs, val_inputs, train_targets, val_targets = train_test_split(
    data["input_ids"], data["labels"], test_size=0.2, random_state=42
)

train_attention_masks, val_attention_masks = train_test_split(
    data["attention_mask"], test_size=0.2, random_state=42
)

train_dataset = Dataset.from_dict({
    "input_ids": train_inputs,
    "attention_mask": train_attention_masks,
    "labels": train_targets,
})

val_dataset = Dataset.from_dict({
    "input_ids": val_inputs,
    "attention_mask": val_attention_masks,
    "labels": val_targets,
})

dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

training_args = TrainingArguments(
    output_dir="./ast-t5-results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

trainer.train()

save_directory = "./fine-tuned-AST-T5"
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

