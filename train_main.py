# semantic_deviation_detection.py
# This script implements a dual-tower model for semantic deviation detection with dynamic thresholding.

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.metrics import roc_curve, f1_score
from transformers import AutoModel, BertTokenizerFast
from tqdm import tqdm
import multiprocessing


class MyDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.samples = []
        self._load_data()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            "label": torch.tensor(self.samples[idx]["label"], dtype=torch.long),
            "data": self.samples[idx]["data"]
        }

    def _load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {os.path.abspath(self.file_path)}")

        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    if isinstance(entry, dict) and 'label' in entry and 'data' in entry:
                        self.samples.append(entry)
                except Exception as e:
                    print(f"Line {line_num} error: {e}")
        print(f"Loaded {len(self.samples)} samples.")


class SemanticComparator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained('albert-tiny-chinese-ws', ignore_mismatched_sizes=True, num_labels=1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(312 * 2, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 1)
        )
        self.threshold = torch.nn.Parameter(torch.tensor(0.0))
        self._init_weights(self.bert.pooler)
        self.bert.pooler.requires_grad_(True)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, groups_input1, groups_input2):
        batch_logits = []
        for sample_input1, sample_input2 in zip(groups_input1, groups_input2):
            input_ids1 = torch.cat([x['input_ids'] for x in sample_input1])
            attention_mask1 = torch.cat([x['attention_mask'] for x in sample_input1])
            emb1 = self.bert(input_ids=input_ids1, attention_mask=attention_mask1).last_hidden_state[:, 0]

            input_ids2 = torch.cat([x['input_ids'] for x in sample_input2])
            attention_mask2 = torch.cat([x['attention_mask'] for x in sample_input2])
            emb2 = self.bert(input_ids=input_ids2, attention_mask=attention_mask2).last_hidden_state[:, 0]

            combined = torch.cat([emb1, emb2], dim=1)
            logits = self.classifier(combined).squeeze()
            batch_logits.append(logits.mean())
        return torch.stack(batch_logits), self.threshold.sigmoid()


class GroupCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = 512

    def __call__(self, batch):
        result = {'groups_input1': [], 'groups_input2': [], 'labels': []}
        for item in batch:
            group_input1, group_input2 = [], []
            for data_item in item['data']:
                enc1 = self.tokenizer(data_item['content2_para'], data_item['matches'][0]['content1_para'],
                                      truncation='longest_first', max_length=self.max_length,
                                      padding='max_length', return_tensors='pt')
                enc2 = self.tokenizer(data_item['content2_para'], data_item['matches'][1]['content1_para'],
                                      truncation='longest_first', max_length=self.max_length,
                                      padding='max_length', return_tensors='pt')
                group_input1.append(enc1)
                group_input2.append(enc2)

            result['groups_input1'].append(group_input1)
            result['groups_input2'].append(group_input2)
            result['labels'].append(item['label'])
        result['labels'] = torch.tensor(result['labels'], dtype=torch.float)
        return result


def calculate_optimal_threshold(all_logits, all_labels, prev_thresholds=None):
    probs = torch.sigmoid(all_logits).numpy()
    labels = all_labels.numpy()
    fpr, tpr, thresholds = roc_curve(labels, probs)
    youden = tpr - fpr
    f1_scores = [f1_score(labels, (probs > t).astype(int)) for t in thresholds]
    combined = 0.6 * np.array(f1_scores) + 0.4 * youden
    idx = np.argmax(combined)
    stop = len(prev_thresholds or []) >= 3 and np.all(np.abs(np.diff(prev_thresholds[-3:])) < 0.01)
    return thresholds[idx], stop


def train():
    config = {"batch_size": 8, "lr": 2e-5, "epochs": 20, "model_save_path": "./best_model.pt"}
    multiprocessing.set_start_method('spawn', force=True)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    train_data = MyDataset('split_data/train.jsonl')
    test_data = MyDataset('split_data/test.jsonl')
    val_size = int(0.2 * len(train_data))
    train_data, val_data = random_split(train_data, [len(train_data) - val_size, val_size])

    collator = GroupCollator(tokenizer)
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], collate_fn=collator, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], collate_fn=collator, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], collate_fn=collator, shuffle=False)

    model = SemanticComparator().to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.bert.pooler.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 2e-4},
        {'params': model.threshold, 'lr': 1e-3}
    ], weight_decay=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    threshold_history, best_acc = [], 0.0

    for epoch in range(config['epochs']):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            inputs1 = [[{k: v.to(device) for k, v in d.items()} for d in s] for s in batch['groups_input1']]
            inputs2 = [[{k: v.to(device) for k, v in d.items()} for d in s] for s in batch['groups_input2']]
            labels = batch['labels'].to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits, _ = model(inputs1, inputs2)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs1 = [[{k: v.to(device) for k, v in d.items()} for d in s] for s in batch['groups_input1']]
                inputs2 = [[{k: v.to(device) for k, v in d.items()} for d in s] for s in batch['groups_input2']]
                labels = batch['labels'].to(device)
                logits, _ = model(inputs1, inputs2)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        best_thresh, stop = calculate_optimal_threshold(all_logits, all_labels, threshold_history)
        threshold_history.append(best_thresh)

        if len(threshold_history) >= 3:
            best_thresh = np.mean(threshold_history[-3:])
        if stop:
            print(f"Early stopping: threshold stabilized at {best_thresh:.4f}")
            break

        with torch.no_grad():
            model.threshold.data = 0.7 * model.threshold.data + 0.3 * torch.tensor(best_thresh).logit()

        preds = (torch.sigmoid(all_logits) > best_thresh).long()
        acc = (preds == all_labels.long()).float().mean().item()
        print(f"Epoch {epoch+1} Accuracy: {acc:.4f} | Threshold: {best_thresh:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                'model_state': model.state_dict(),
                'threshold': model.threshold.data,
                'threshold_history': threshold_history
            }, config['model_save_path'])
            print(f"Model saved with accuracy: {acc:.4f}")

    print("Testing model...")
    checkpoint = torch.load(config['model_save_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    best_thresh = np.mean(checkpoint['threshold_history'][-3:])
    model.threshold.data = torch.tensor(best_thresh).logit().to(device)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Test]"):
            inputs1 = [[{k: v.to(device) for k, v in d.items()} for d in s] for s in batch['groups_input1']]
            inputs2 = [[{k: v.to(device) for k, v in d.items()} for d in s] for s in batch['groups_input2']]
            labels = batch['labels'].to(device)
            logits, threshold = model(inputs1, inputs2)
            preds = (torch.sigmoid(logits) > threshold).long()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Final Test Accuracy: {correct / total:.4f}")


if __name__ == '__main__':
    train()
