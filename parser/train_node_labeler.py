import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import classification_report
import os
import json

# Define constants
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 4096
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.01

# Define dataset class
class NodeLabelingDataset(Dataset):
    def __init__(self, data, labels, tokenizer, max_length, source_order=['question', 'response']):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = self.create_label_map(labels)
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.data = self.convert_to_bi_labels(data, source_order)
        
    def create_label_map(self, all_labels):
        unique_labels = set()
        for label_list in all_labels:
            unique_labels.update(label_list)
        
        # Create B- and I- prefixes for each label
        label_map = {'O': 0}  # Outside tag
        i = 1
        for label in sorted(unique_labels):
            label_map[f"B-{label}"] = i
            i += 1
            label_map[f"I-{label}"] = i
            i += 1
        
        return label_map
        
    def __len__(self):
        return len(self.data)
    
    def convert_to_bi_labels(self, data, source_order):
        """
        Convert character spans and labels to BIO tagging scheme at token level
        """
        final_data = []

        for datum in data:
            text = ""
            source_offset = {o: None for o in source_order}
            for o in source_order:
                text += f"[{o.upper()}]\n"
                source_offset[o] = len(text)
                text += datum["raw_text"][o] + "\n\n"

            tokenized = self.tokenizer(text, return_offsets_mapping=True, 
                                   truncation=True, max_length=self.max_length)
        
            # Initialize with O tags
            token_labels = ['O'] * len(tokenized['offset_mapping'])
            
            # Skip special tokens
            for i, (start, end) in enumerate(tokenized['offset_mapping']):
                if start == 0 and end == 0:  # Special token
                    token_labels[i] = -100  # Ignore in loss computation
            
            # Assign labels to tokens
            for span_idx, node in enumerate(datum["nodes"]):
                label_assigned = False
                offset = source_offset[node["source"]]
                start_char, end_char = node["start"] + offset, node["end"] + offset
                label = node["label"]
                
                for i, (start, end) in enumerate(tokenized['offset_mapping']):
                    # Skip special tokens
                    if start == 0 and end == 0:
                        continue
                    
                    # Check if this token overlaps with the span
                    if end <= start_char or start >= end_char:
                        continue
                    
                    # Determine if this is a beginning or inside token
                    if start_char <= start:
                        token_labels[i] = f"B-{label}"
                        label_assigned = True
                    else:
                        token_labels[i] = f"I-{label}"
                        label_assigned = True
                        
            # Convert string labels to IDs, with -100 for ignored positions
            token_label_ids = []
            for label in token_labels:
                if label == -100:
                    token_label_ids.append(-100)
                else:
                    token_label_ids.append(self.label2id.get(label, self.label2id['O']))
            
            assert len(tokenized["input_ids"]) == len(token_label_ids), \
                f"Input IDs and token labels length mismatch: {len(tokenized['input_ids'])} vs {len(token_label_ids)}"
            # append to final data
            final_data.append({
                "input_ids": torch.tensor(tokenized["input_ids"]), # input
                "attention_mask": torch.tensor(tokenized["attention_mask"]), # attention mask
                "labels": torch.tensor(token_label_ids, dtype=torch.long) # output
            })

        return final_data
    
    def __getitem__(self, idx):
        return self.data[idx]

def node_collate_fn(batch):
    # batch: List[Tuple[Dict, torch.Tensor]]
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item["labels"] for item in batch]
    # Pad sequences to the maximum length in the batch
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels
    }

def train_model(model, train_dataloader, optimizer, scheduler, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Average loss: {avg_loss:.4f}")
    
    return model

def evaluate_model(model, eval_dataloader, id2label, device):
    model.eval()
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)
            
            # Convert to labels and filter out ignored indices (-100)
            for i in range(len(labels)):
                true_label = labels[i].cpu().numpy()
                pred = predictions[i].cpu().numpy()
                
                for j in range(len(true_label)):
                    if true_label[j] != -100:
                        all_true_labels.append(true_label[j])
                        all_predictions.append(pred[j])
    
    # Convert numeric labels back to text labels for reporting
    true_labels_text = [id2label[label] for label in all_true_labels]
    pred_labels_text = [id2label[label] for label in all_predictions]
    
    # Print classification report
    print(classification_report(true_labels_text, pred_labels_text))
    
    return all_predictions, all_true_labels

def main_train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # load data
    data = []
    for file in os.listdir("web/data"):
        if file.endswith(".json"):
            with open(os.path.join("web/data", file), "r") as f:
                datum = json.load(f)
                if len(datum["nodes"]) <= len(datum["edges"]):
                    continue
                data.append(datum)
    print(f"Loaded {len(data)} data samples.")

    # load labels
    with open("web/static/labels.json", "r") as f:
        labels = json.load(f)["node_labels"]
    
    # Create dataset
    dataset = NodeLabelingDataset(data, labels, tokenizer, MAX_LENGTH)
    
    # Split data into train and validation (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=node_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=node_collate_fn)
    
    # Initialize model
    num_labels = len(dataset.label2id)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=num_labels,
        id2label=dataset.id2label,
        label2id=dataset.label2id
    )
    model.require_grad=False
    model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=WARMUP_STEPS, 
        num_training_steps=total_steps
    )
    
    # Train the model
    model = train_model(model, train_dataloader, optimizer, scheduler, device, EPOCHS)
    
    # Evaluate the model
    predictions, true_labels = evaluate_model(model, val_dataloader, dataset.id2label, device)
    
    # Save the model
    model.save_pretrained("./qwen_sequence_labeling_model")
    tokenizer.save_pretrained("./qwen_sequence_labeling_model")
    
    print("Model training and evaluation completed!")

# Example function to use the trained model for prediction
def predict_spans(model, tokenizer, text, id2label, device):
    model.eval()
    
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = inputs.pop("offset_mapping")
    
    # Get token predictions
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    # Convert token predictions to spans
    predicted_spans = []
    current_label = None
    start_idx = None
    
    for i, (offset, pred) in enumerate(zip(offset_mapping[0], predictions[0])):
        token_start, token_end = offset.tolist()
        pred_label = id2label[pred.item()]
        
        # Skip special tokens and "O" labels
        if token_start == token_end == 0 or pred_label == "O":
            if current_label:
                predicted_spans.append((start_idx, token_start, current_label.replace("B-", "").replace("I-", "")))
                current_label = None
            continue
        
        # Start of a new entity
        if pred_label.startswith("B-"):
            if current_label:
                predicted_spans.append((start_idx, token_start, current_label.replace("B-", "").replace("I-", "")))
            
            current_label = pred_label
            start_idx = token_start
        
        # Continuation of current entity
        elif pred_label.startswith("I-") and current_label and pred_label.replace("I-", "") == current_label.replace("B-", "").replace("I-", ""):
            continue
        
        # End of sequence or different I- tag
        else:
            if current_label:
                predicted_spans.append((start_idx, token_start, current_label.replace("B-", "").replace("I-", "")))
            
            if pred_label.startswith("I-"):
                # This is an error case, I- without preceding B-
                current_label = "B-" + pred_label.replace("I-", "")
                start_idx = token_start
            else:
                current_label = None
    
    # Add the last span if exists
    if current_label:
        predicted_spans.append((start_idx, len(text), current_label.replace("B-", "").replace("I-", "")))
    
    return predicted_spans

if __name__ == "__main__":
    main_train()