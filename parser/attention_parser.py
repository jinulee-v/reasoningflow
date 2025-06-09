import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from sklearn.metrics import classification_report
import os
import json

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

# Define constants
# MODEL_NAME = "Qwen/Qwen2.5-3B"
MODEL_NAME = "answerdotai/ModernBERT-large"
MAX_LENGTH = 4096
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 6e-5
WARMUP_STEPS = 10
WEIGHT_DECAY = 0.003

index_I = 1 # index of "I" label in node_label2id
index_no_edge = 0 # index of "no_edge" label in edge_label2id

# Define dataset class
class ReasoningFlowDataset(Dataset):

    def __init__(self, data, node_labels, edge_labels, tokenizer, max_length, source_order=['question', 'response'], binary_edge_labels=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.node_label2id = self.create_node_label_map(node_labels)
        self.node_id2label = [k for k, v in self.node_label2id.items()]
        assert self.node_id2label[index_I] == "I", f"Index of 'I' label should be {index_I}, but got {self.node_id2label[index_I]}"

        self.binary_edge_labels = binary_edge_labels
        if binary_edge_labels:
            self.edge_id2label = ["no_edge", "edge"]
            self.edge_label2id = {v: k for k, v in enumerate(["no_edge", "edge"])}
        else:
            self.edge_id2label = ["no_edge"] + edge_labels
            self.edge_label2id = {v: k for k, v in enumerate(["no_edge"] + edge_labels)}
        assert self.edge_id2label[index_no_edge] == "no_edge", f"Index of 'no_edge' label should be {index_no_edge}, but got {self.node_id2label[index_no_edge]}"
            
        self.data = self.convert_to_bi_labels(data, source_order)
        
    def create_node_label_map(self, labels):
        # Create B- and I- prefixes for each label
        label_map = {'O': 0, 'I': 1}  # Outside tag
        i = 2
        for label in sorted(labels):
            label_map[f"B-{label}"] = i
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
            # print(datum["doc_id"])
            text = ""
            source_offset = {o: None for o in source_order}
            for o in source_order:
                text += f"[{o.upper()}]\n"
                source_offset[o] = len(text)
                text += datum["raw_text"][o] + "\n\n"

            tokenized = self.tokenizer(text, return_offsets_mapping=True, truncation=False)
        
            # Initialize with O tags
            token_labels = ['O'] * len(tokenized['offset_mapping'])
            
            # sort datum["nodes"] by start position
            datum["nodes"].sort(key=lambda x: (x["start"] + source_offset[x["source"]]))

            # Skip special tokens
            span_idx = 0
            node_start_positions_dict = {} # maps node IDs to B-label token indices for edge purpose
            node_index_dict = {node["id"]: i for i, node in enumerate(datum["nodes"])}
            for i, (start, end) in enumerate(tokenized['offset_mapping']):
                if start == 0 and end == 0:  # Special token
                    token_labels[i] = -100  # Ignore in loss computation
                # print(start, end, json.dumps(text[start:end]), datum["nodes"][span_idx]["start"] + source_offset[datum["nodes"][span_idx]["source"]], datum["nodes"][span_idx]["end"] + source_offset[datum["nodes"][span_idx]["source"]])
                if end < datum["nodes"][span_idx]["start"] + source_offset[datum["nodes"][span_idx]["source"]]:
                    continue
                if end > datum["nodes"][span_idx]["end"] + source_offset[datum["nodes"][span_idx]["source"]]:
                    span_idx += 1
                if span_idx >= len(datum["nodes"]):
                    span_idx = len(datum["nodes"]) - 1

                span_start = datum["nodes"][span_idx]["start"] + source_offset[datum["nodes"][span_idx]["source"]]
                span_end = datum["nodes"][span_idx]["end"] + source_offset[datum["nodes"][span_idx]["source"]]

                # Determine if this is a beginning or inside token
                label = datum["nodes"][span_idx]["label"]
                label_assigned = False
                if start <= span_start:
                    token_labels[i] = f"B-{label}"
                    label_assigned = True
                    node_start_positions_dict[datum["nodes"][span_idx]["id"]] = i
                else:
                    # token_labels[i] = f"I-{label}"
                    token_labels[i] = f"I"
                    label_assigned = True
                # print("->", token_labels[i])
                assert label_assigned, f"Label not assigned for token {i} in {text[start:end]}"
                        
            # Convert string labels to IDs, with -100 for ignored positions
            token_label_ids = []
            for label in token_labels:
                if label == -100:
                    token_label_ids.append(-100)
                else:
                    token_label_ids.append(self.node_label2id.get(label, self.node_label2id['O']))
            
            assert len(tokenized["input_ids"]) == len(token_label_ids), \
                f"Input IDs and token labels length mismatch: {len(tokenized['input_ids'])} vs {len(token_label_ids)}"
            
            # Add edge labels
            # Default: Upper eschelon is no edge, lower eschelon is -100
            # print(len(node_start_positions_dict), len(node_index_dict))
            edge_labels = torch.ones(len(node_start_positions_dict), len(node_start_positions_dict), dtype=torch.long) * -100
            for i in range(len(node_start_positions_dict) - 1):
                for j in range(i + 1, len(node_start_positions_dict)):
                        edge_labels[i, j] = self.edge_label2id.get("no_edge")

            for edge in datum["edges"]:
                if edge["label"].strip() == "":
                    continue
                start_id = node_index_dict.get(edge["from_node_id"], None)
                end_id = node_index_dict.get(edge["to_node_id"], None)
                assert start_id < end_id, f"Edge from {edge['from_node_id']} to {edge['to_node_id']} is not in order"
                if start_id is not None and end_id is not None:
                    if self.binary_edge_labels:
                        edge_labels[start_id, end_id] = self.edge_label2id.get("edge")
                    else:
                        # For non-binary edge labels, we use the label directly
                        edge_labels[start_id, end_id] = self.edge_label2id.get(edge["label"])

            # append to final data
            final_data.append({
                "input_ids": torch.tensor(tokenized["input_ids"]), # input
                "attention_mask": torch.tensor(tokenized["attention_mask"]), # attention mask
                "node_labels": torch.tensor(token_label_ids, dtype=torch.long), # node labels
                "edge_labels": edge_labels, # edge labels
                "node_start_positions": list(node_start_positions_dict.values()) # node start positions
            })

        return final_data
    
    def __getitem__(self, idx):
        return self.data[idx]

class ReasoningFlowLabeler(nn.Module):
    def __init__(self, model_name, node_labels, edge_labels):
        super(ReasoningFlowLabeler, self).__init__()
        self.model = AutoModel.from_pretrained(
            model_name
        )
        self.node_labels = node_labels
        self.edge_labels = edge_labels
        self.node_labeling_layer = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.node_labeling_head = nn.Linear(self.model.config.hidden_size, len(node_labels))
        # self.edge_labeling_head = nn.Bilinear(self.model.config.hidden_size, self.model.config.hidden_size, len(edge_labels))
        self.edge_labeling_query_layer = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.edge_labeling_key_layer = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.edge_labeling_head_weight = nn.Parameter(torch.empty(len(edge_labels), self.model.config.hidden_size, self.model.config.hidden_size))
        # self.edge_labeling_head_weight = nn.Parameter(torch.empty(len(edge_labels), self.model.config.hidden_size, self.model.config.hidden_size))
        self.edge_labeling_head_bias = nn.Parameter(torch.zeros(len(edge_labels)))
        nn.init.xavier_uniform_(self.edge_labeling_head_weight)
        nn.init.zeros_(self.edge_labeling_head_bias)
        self.dtype = torch.bfloat16
    
    def forward(self, input_ids, attention_mask, node_start_positions=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        num_tokens = hidden_states.size(1)  # Get number of tokens
        
        # Node labeling
        # node_logits = self.node_labeling_head(torch.relu(self.node_labeling_layer(hidden_states))) # num_tokens x num_labels
        node_logits = self.node_labeling_head(torch.relu(self.node_labeling_layer(hidden_states)))  # (batch_size, seq_len, hidden_dim)
        
        # Edge labeling
        if node_start_positions is None:
            # If no node start positions are provided,
            # Use node_logits to determine the predicted start positions
            pass

        num_nodes = node_start_positions.size(1)
        # print(hidden_states.shape, node_start_positions)
        # node_start_positions: (batch_size, num_nodes)
        # hidden_states: (batch_size, seq_len, hidden_dim)
        # We want to gather hidden_states at node_start_positions for each batch
        # Output: (batch_size, num_nodes, hidden_dim)
        node_start_hidden_states = torch.gather(
            hidden_states,
            1,
            node_start_positions.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        )
        # print(hidden_states.shape, node_start_hidden_states.shape, num_nodes)

        # node_hidden: (B, N, D)
        left = torch.relu(self.edge_labeling_key_layer(node_start_hidden_states))   # (B, N, D)
        right = torch.relu(self.edge_labeling_query_layer(node_start_hidden_states))  # (B, N, D)

        # einsum for bilinear scoring: (B, N, D) * (C, D, D') * (B, N', D) -> (B, N, N', C) # D==D', N==N'
        edge_logits = torch.einsum('bnd,cde,bme->bnmc', left, self.edge_labeling_head_weight, right)
        # normalize by sqrt of hidden size
        edge_logits = edge_logits / (self.model.config.hidden_size ** 0.5)
        edge_logits += self.edge_labeling_head_bias  # Add bias for each edge label
        
        # edge_logits = self.edge_labeling_head(
        #     node_start_hidden_states.unsqueeze(2).expand((-1,-1,num_nodes,-1)).reshape(-1, hidden_states.size(-1)),
        #     node_start_hidden_states.unsqueeze(1).expand((-1,num_nodes,-1,-1)).reshape(-1, hidden_states.size(-1))
        # )

        # print("edge_logits:", edge_logits.shape)
        edge_logits = edge_logits.view(-1, num_nodes, num_nodes, len(self.edge_labels))  # Reshape to (batch_size, num_tokens, num_tokens, num_edge_labels)
        # print(edge_logits.shape, node_start_positions.shape)
        edge_logits *= (node_start_positions != 0).unsqueeze(2).unsqueeze(3).to(edge_logits.dtype)

        return {
            'node_logits': torch.log_softmax(node_logits, dim=-1),  # Node logits
            'edge_logits': torch.log_softmax(edge_logits, dim=-1)  # Edge logits
        }

def node_collate_fn(batch):
    # batch: List[Tuple[Dict, torch.Tensor]]
    # Pad sequences to the maximum length in the batch

    input_ids = [item['input_ids'] for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)

    attention_masks = [item['attention_mask'] for item in batch]
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    node_labels = [item["node_labels"] for item in batch]
    node_labels = torch.nn.utils.rnn.pad_sequence(node_labels, batch_first=True, padding_value=-100)

    # to pad edge labels, we need to copy it to the largest shape (1, max_len, max_len) then concatenate
    max_len = max([x["edge_labels"].shape[0] for x in batch]) # get largest dimension
    padded_tensors = []
    for t in [item["edge_labels"] for item in batch]:
        h, w = t.size()
        pad_h = max_len - h
        pad_w = max_len - w
        # Padding is (left, right, top, bottom)
        padded = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode='constant', value=-100)
        padded_tensors.append(padded.unsqueeze(0))  # Add batch dimension
    edge_labels = torch.cat(padded_tensors, dim=0)

    # Get node start positions
    node_start_positions = [item["node_start_positions"] for item in batch]
    max_node_start_len = max(len(pos) for pos in node_start_positions)
    node_start_positions = [pos + [0] * (max_node_start_len - len(pos)) for pos in node_start_positions]
    node_start_positions = torch.tensor(node_start_positions, dtype=torch.long)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'node_labels': node_labels,
        'edge_labels': edge_labels,
        'node_start_positions': node_start_positions
    }

def train_model(model: ReasoningFlowLabeler, train_dataloader, optimizer, scheduler, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            node_labels = batch['node_labels'].to(device)
            edge_labels = batch['edge_labels'].to(device)
            node_start_positions = batch['node_start_positions'].to(device)
            
            # Define class weights
            node_class_weights = torch.tensor([1.0] * len(model.node_labels)).to(device)
            node_class_weights[index_I] = 0.1 # Disregard "I" labels; TODO replace hard-coded 1 to the index of "I" label
            edge_class_weights = torch.tensor([1.0] * len(model.edge_labels)).to(device)
            edge_class_weights[index_no_edge] = 0.1 # Disregard "no_edge" labels; TODO replace hard-coded 0 to the index of "no_edge" label

            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                node_start_positions=node_start_positions
            )
            
            node_logits = outputs["node_logits"]
            # print(f"Node logits shape: {node_logits.shape}, Labels shape: {labels.shape}, weight shape: {class_weights.shape}")
            node_loss_fn = torch.nn.CrossEntropyLoss(weight=node_class_weights, ignore_index=-100)
            node_loss = node_loss_fn(node_logits.view(-1, len(model.node_labels)), node_labels.view(-1))
            
            edge_logits = outputs["edge_logits"]
            edge_loss_fn = torch.nn.CrossEntropyLoss(weight=edge_class_weights, ignore_index=-100)
            # print(f"Edge logits shape: {edge_logits.shape}, Edge labels shape: {edge_labels.shape}")
            edge_loss = edge_loss_fn(edge_logits.view(-1, len(model.edge_labels)), edge_labels.view(-1))

            loss = node_loss + edge_loss
            # loss = node_loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Average loss: {avg_loss:.4f}")
    
    return model

def evaluate_model(model, eval_dataloader, node_id2label, edge_id2label, device):
    model.eval()
    node_all_predictions = []
    node_all_true_labels = []
    edge_all_predictions = []
    edge_all_true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            node_labels = batch['node_labels'].to(device)
            edge_labels = batch['edge_labels'].to(device)
            node_start_positions = batch['node_start_positions'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                node_start_positions=node_start_positions
            )
            
            # node logits
            node_logits = outputs["node_logits"]
            predictions = torch.argmax(node_logits, dim=2)
            
            # Convert to labels and filter out ignored indices (-100)
            for i in range(len(node_labels)):
                true_label = node_labels[i].cpu().numpy()
                pred = predictions[i].cpu().numpy()
                cnt = 0
                for j in range(len(true_label)):
                    if true_label[j] != -100:
                        cnt += 1
                        node_all_true_labels.append(true_label[j])
                        node_all_predictions.append(pred[j])
            
            # edge logits
            edge_logits = outputs["edge_logits"]
            edge_predictions = torch.argmax(edge_logits, dim=3)

            # Convert to labels and filter out ignored indices (-100)
            for i in range(len(edge_labels)):
                true_label = edge_labels[i].cpu().numpy()
                pred = edge_predictions[i].cpu().numpy()
                
                for j in range(len(true_label)):
                    for k in range(len(true_label)):
                        if true_label[j, k] != -100:
                            edge_all_true_labels.append(true_label[j, k])
                            edge_all_predictions.append(pred[j, k])
    
    # Convert numeric labels back to text labels for reporting
    node_true_labels_text = [node_id2label[label] for label in node_all_true_labels]
    node_pred_labels_text = [node_id2label[label] for label in node_all_predictions]

    # Print segmentation example
    print("Segmentation example:")
    for i in range(len(node_true_labels_text)):
        if node_id2label[node_all_true_labels[i]].startswith("B-"):
            print(f"True: {node_id2label[node_all_true_labels[i]]}, Pred: {node_id2label[node_all_predictions[i]]}")
    
    # Print classification report
    print(classification_report(node_true_labels_text, node_pred_labels_text))

    total_mistake = 0
    bi_mistake = 0
    for t, p in zip(node_true_labels_text, node_pred_labels_text):
        if t.startswith("B-") != p.startswith("B-"):
            bi_mistake += 1 # when model predicts B- but true label is I- or O, or vice versa
        if t != p:
            total_mistake += 1
    print(f"Total mistakes: {total_mistake}, B-I mistakes: {bi_mistake}")

    # Print edge classification report
    edge_true_labels_text = [edge_id2label[label] for label in edge_all_true_labels]
    edge_pred_labels_text = [edge_id2label[label] for label in edge_all_predictions]
    print("Edge classification report:")
    print(classification_report(edge_true_labels_text, edge_pred_labels_text, zero_division=0))

    # Binary prediction: there is an edge or not
    edge_binary_true = [1 if label != "no_edge" else 0 for label in edge_true_labels_text]
    edge_binary_pred = [1 if label != "no_edge" else 0 for label in edge_pred_labels_text]
    print("Edge classification report (binary):")
    print(classification_report(edge_binary_true, edge_binary_pred, zero_division=0))
    
    return node_all_predictions, node_all_true_labels, edge_all_predictions, edge_all_true_labels

def main_train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("<Load tokenizer...>")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # load data
    print("<Load data...>")
    data = []
    for file in os.listdir("web/data"):
        if file.endswith(".json"):
            with open(os.path.join("web/data", file), "r") as f:
                datum = json.load(f)
                if len(datum["nodes"]) > len(datum["edges"]):
                    continue
                data.append(datum)
    print(f"Loaded {len(data)} data samples.")

    # load labels
    print("<Load labels...>")
    with open("web/static/labels.json", "r") as f:
        labels_raw = json.load(f)
        node_labels = labels_raw["node_labels"]
        edge_labels = labels_raw["edge_labels"]
    
    # Create dataset
    print("<Preprocess dataset...>")
    # dataset = ReasoningFlowDataset(data, node_labels, edge_labels, tokenizer, MAX_LENGTH)
    dataset = ReasoningFlowDataset(data, node_labels, edge_labels, tokenizer, MAX_LENGTH, binary_edge_labels=True)
    
    # Split data into train and validation (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # train_dataset = val_dataset = dataset
    # Train data label distribution
    for i in range(len(dataset.node_label2id)):
        cnt = 0
        for j in range(len(train_dataset)):
            cnt += sum(train_dataset[j]["node_labels"] == i)
        print(f"Label {dataset.node_id2label[i]}: {cnt} samples")
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=node_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=node_collate_fn)
    
    # Initialize model
    print("<Initialize ReasonigFlowLabeler...>")
    num_labels = len(dataset.node_label2id)
    model = ReasoningFlowLabeler(
        MODEL_NAME, 
        node_labels=dataset.node_id2label, 
        edge_labels=dataset.edge_id2label
    )
    model.to(device)
    
    # # Apply LoRA
    # peft_config = LoraConfig(
    #     task_type=TaskType.TOKEN_CLS,
    #     inference_mode=False,
    #     r=32,
    #     lora_alpha=32,
    #     lora_dropout=0.1,
    #     target_modules=[
    #         "q_proj", "k_proj", "v_proj",  # Attention projections
    #         "o_proj",                      # Output projection in attention
    #         "gate_proj", "up_proj", "down_proj"  # MLP layers
    #     ],
    # )
    # model = get_peft_model(model, peft_config)
    
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
    node_predictions, node_true_labels, edge_predictions, edge_true_labels = evaluate_model(model, val_dataloader, dataset.node_id2label, dataset.edge_id2label, device)
    
    # Save the model
    # model = model.merge_and_unload()  # Merge LoRA weights back into the base model
    # model.save_pretrained("./qwen_sequence_labeling_model")
    # tokenizer.save_pretrained("./qwen_sequence_labeling_model")
    torch.save(model.state_dict(), "./reasoning_flow_labeler.pth")
    
    print("Model training and evaluation completed!")

# Example function to use the trained model for prediction
def predict_spans(model, tokenizer, text, node_id2label, device):
    model.eval()
    
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = inputs.pop("offset_mapping")
    
    # Get token predictions
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)["node_logits"]
        predictions = torch.argmax(outputs, dim=2)
    
    # Convert token predictions to spans
    predicted_spans = []
    current_label = None
    start_idx = None
    
    for i, (offset, pred) in enumerate(zip(offset_mapping[0], predictions[0])):
        token_start, token_end = offset.tolist()
        pred_label = node_id2label[pred.item()]
        
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
        elif pred_label == "I":
            continue
    
    # Add the last span if exists
    if current_label:
        predicted_spans.append((start_idx, len(text), current_label.replace("B-", "").replace("I-", "")))
    
    return predicted_spans

def main_predict(): # Only predict spans, since edge quality is very low
    # Load the model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    try:
        model = ReasoningFlowLabeler(
            MODEL_NAME, 
            node_labels=["O", "I", "B-Example"],  # Example labels
            edge_labels=["no_edge", "edge"]
        )
    except:
        print("<Load labels...>")
        with open("web/static/labels.json", "r") as f:
            labels_raw = json.load(f)
            node_labels = labels_raw["node_labels"]
            edge_labels = labels_raw["edge_labels"]
        model = ReasoningFlowLabeler(
            MODEL_NAME, 
            node_labels=["O", "I", "B-Example"],  # Example labels
            edge_labels=edge_labels
        )
    model.load_state_dict(torch.load("./reasoning_flow_labeler.pth"))
    model.to(device)
    
    # Load the data
    print("<Load data...>")
    data = []
    for file in os.listdir("scp116k_data"):
        if file.endswith(".json"):
            with open(os.path.join("scp116k_data", file), "r") as f:
                datum = json.load(f)
                # if len(datum["nodes"]) > len(datum["edges"]):
                #     continue
                data.append(datum)
    print(f"Loaded {len(data)} data samples.")
    
    # Predict spans
    for datum in data:
        text = datum["text"]
        predicted_spans = predict_spans(model, tokenizer, text, model.node_id2label, device)
        print(f"Predicted spans for {text}: {predicted_spans}")

if __name__ == "__main__":
    # # Run the training process
    main_train()
    # main_predict()