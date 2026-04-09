import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, get_scheduler
from torch.optim import AdamW
import pandas as pd
import numpy as np
import os
from rdkit import Chem
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SMILES randomization for data augmentation, n is number of randomized (non-canonical) output SMIs
def randomize_smiles(smiles, n=4):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    randomized = set()
    while len(randomized) < n:
        randomized.add(Chem.MolToSmiles(mol, canonical=False, doRandom=True))
    return list(randomized)

# Dataset class 
class SmilesDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer, max_length=128, augment=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.smiles_data = []
        self.labels = []

        for smi, label in zip(smiles_list, labels):
            smiles_ensemble = [smi]
            if augment:
                smiles_ensemble += randomize_smiles(smi, n=4)
            for s in smiles_ensemble:
                self.smiles_data.append(s)
                self.labels.append(label)

    def __len__(self):
        return len(self.smiles_data)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.smiles_data[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# Model with attention pooling and dropout
class ChemBERTaForRegression(nn.Module):
    def __init__(self, pretrained_dir, dropout_prob=0.2):
        super().__init__()
        self.chemberta = RobertaModel.from_pretrained(pretrained_dir)
        hidden_size = self.chemberta.config.hidden_size
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_prob)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
        attn_scores = self.attention(hidden_states).squeeze(-1)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(hidden_states * attn_weights.unsqueeze(-1), dim=1)
        pooled = self.dropout(pooled)
        output = self.regressor(pooled).squeeze(-1)
        return output

# Load data
data = pd.read_csv('./labeled_data_t1.csv') 
smiles = data['SMILES'].tolist()
labels = data['t1'].to_numpy()  # T1 labels

# Train/val split
smiles_train, smiles_val, labels_train, labels_val = train_test_split(smiles, labels, test_size=0.1, random_state=42)

# Tokenizer
pretrained_dir = os.path.expanduser("PATH_TO/chemberta_380k/") # the pretrained chemberta 
tokenizer = RobertaTokenizer.from_pretrained(pretrained_dir)

# Datasets
train_dataset = SmilesDataset(smiles_train, labels_train, tokenizer, augment=True)
val_dataset = SmilesDataset(smiles_val, labels_val, tokenizer, augment=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Initialize model
model = ChemBERTaForRegression(pretrained_dir).to(device)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 10
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Loss function
criterion = nn.MSELoss()

# Training loop
def train_epoch(model, loader):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['labels'].to(device)
        preds = model(input_ids, attention_mask)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(loader.dataset)

def eval_epoch(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)
            preds = model(input_ids, attention_mask)
            loss = criterion(preds, targets)
            total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(loader.dataset)

# Train
loss_log=[]
best_val_loss = float('inf')
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss = eval_epoch(model, val_loader)
    print(f"Epoch {epoch+1}/{num_epochs} — Train Loss: {train_loss:.4f} — Val Loss: {val_loss:.4f}")
    loss_log.append({'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss})
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "chemberta_finetuned_t1_state.pt")
        print(f"Best model saved at epoch {epoch+1} with val_loss = {val_loss:.4f}")
pd.DataFrame(loss_log).to_csv("loss_log_t1.csv", index=False)
print("Fine-tuning complete!")


# Predictions for mol_library.smi

# Load mol_library.smi
smiles_list, names_list = [], []
with open("mol_library.smi", "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        smi, name = parts[0], parts[1]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        smiles_list.append(Chem.MolToSmiles(mol, canonical=True))
        names_list.append(name)

# Minimal dataset class for prediction
class PredictDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_length=128):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smi = self.smiles_list[idx]
        enc = self.tokenizer(smi, max_length=self.max_length, padding='max_length',
                             truncation=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0)}

pred_dataset = PredictDataset(smiles_list, tokenizer)
pred_loader = DataLoader(pred_dataset, batch_size=16)

# Run predictions
model.eval()
torch.set_grad_enabled(False)
preds = []
for batch in pred_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    output = model(input_ids, attention_mask)
    preds.extend(output.squeeze(-1).cpu().numpy())

# Save predictions
df = pd.DataFrame({
    "name": names_list,
    "smiles": smiles_list,
    "t1_pred": preds
})
df.to_csv("t1_predictions.csv", index=False)
print("Predictions saved to t1_predictions.csv")

