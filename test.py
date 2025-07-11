import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, f1_score, precision_score, recall_score, confusion_matrix, \
    ConfusionMatrixDisplay, classification_report, auc
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
from transformers import AlbertModel, AlbertTokenizerFast, BertModel, BertTokenizerFast, AutoModel
import json
import multiprocessing


class MyDataset(Dataset):
    def __init__(self, file_path):
        """
        改进后的数据集类，确保文件单次打开
        """
        self.file_path = file_path
        self.samples = []  # 存储全部数据在内存中
        self._load_data()  # 初始化时一次性加载数据

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        直接从内存读取数据，无需文件操作
        """
        return {
            "label": torch.tensor(self.samples[idx]["label"], dtype=torch.long),
            "data": self.samples[idx]["data"]
        }

    def _load_data(self):
        """ 单次打开文件并完全加载到内存 """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"文件不存在: {os.path.abspath(self.file_path)}")

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                buffer = []  # 临时缓冲区
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                        # 数据校验
                        if not (isinstance(entry, dict)
                                and 'label' in entry
                                and 'data' in entry
                                and isinstance(entry['data'], list)):
                            raise ValueError("数据结构错误")
                        buffer.append(entry)
                    except Exception as e:
                        print(f"行 {line_num} 解析失败: {str(e)}")
                        continue

                # 批量存储到内存
                self.samples = buffer
                print(f"成功加载 {len(self.samples)} 条数据")

        except IOError as e:
            print(f"文件读取失败: {str(e)}")
            raise


class SemanticComparator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained('albert-tiny-chinese-ws', ignore_mismatched_sizes=True,
    num_labels=1)


        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(312 * 2, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 1)
        )

        # 动态阈值参数（新增）
        self.threshold = torch.nn.Parameter(torch.tensor(0.0))  # 初始化为logit(0.5)

        self._init_weights(self.bert.pooler)

        self.bert.pooler.requires_grad_(True)

    def _init_weights(self, module):
        """精确复制ALBERT的初始化方法"""
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1)
            if module.bias is not None:
                module.bias.data.zero_()
    def forward(self, groups_input1, groups_input2):
        batch_logits = []

        # 处理每个样本的数据项
        for sample_idx in range(len(groups_input1)):
            sample_input1 = groups_input1[sample_idx]
            sample_input2 = groups_input2[sample_idx]

            input_ids = torch.cat([x['input_ids'] for x in sample_input1])
            attention_mask = torch.cat([x['attention_mask'] for x in sample_input1])
            outputs1 = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask).last_hidden_state[:, 0]

            input_ids = torch.cat([x['input_ids'] for x in sample_input2])
            attention_mask = torch.cat([x['attention_mask'] for x in sample_input2])
            outputs2 = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask).last_hidden_state[:, 0]

            combined = torch.cat([outputs1, outputs2], dim=1)
            logits = self.classifier(combined).squeeze()

            batch_logits.append(logits.mean())

        return torch.stack(batch_logits), self.threshold.sigmoid()


class GroupCollator:
    """支持组处理的设备感知collator"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = 512

    def __call__(self, batch):
        processed = {
            'groups_input1': [],
            'groups_input2': [],
            'labels': []
        }

        for item in batch:
            group_input1 = []
            group_input2 = []

            # 处理单个样本内的所有data项
            for data_item in item['data']:
                # 处理第一个匹配对
                enc1 = self.tokenizer(
                    data_item['content2_para'],
                    data_item['matches'][0]['content1_para'],
                    truncation='longest_first',
                    max_length=self.max_length,
                    padding='max_length',
                    return_tensors='pt'
                )

                # 处理第二个匹配对
                enc2 = self.tokenizer(
                    data_item['content2_para'],
                    data_item['matches'][1]['content1_para'],
                    truncation='longest_first',
                    max_length=self.max_length,
                    padding='max_length',
                    return_tensors='pt'
                )

                group_input1.append(enc1)
                group_input2.append(enc2)

            processed['groups_input1'].append(group_input1)
            processed['groups_input2'].append(group_input2)
            processed['labels'].append(item['label'])

        return {
            'groups_input1': processed['groups_input1'],
            'groups_input2': processed['groups_input2'],
            'labels': torch.tensor(processed['labels'], dtype=torch.float)
        }


def calculate_optimal_threshold(all_logits, all_labels, prev_thresholds=None):
    probs = torch.sigmoid(all_logits).numpy()
    labels = all_labels.numpy()

    # 基础指标
    fpr, tpr, thresholds = roc_curve(labels, probs)
    youden = tpr - fpr
    f1_scores = []
    for thresh in thresholds:
        preds = (probs > thresh).astype(int)
        f1_scores.append(f1_score(labels, preds))

    # F1 60% + Youden 40% 的综合指标
    combined_scores = 0.6 * np.array(f1_scores) + 0.4 * youden
    optimal_idx = np.argmax(combined_scores)

    # 早停检测（最近3次变化小于1%）
    stop_flag = False
    if prev_thresholds and len(prev_thresholds) >= 3:
        changes = np.abs(np.diff(prev_thresholds[-3:]))
        if np.all(changes < 0.01):
            stop_flag = True

    return thresholds[optimal_idx], stop_flag



def test():
    config = {
        "batch_size": 8,
        "lr": 2e-5,
        "epochs": 20,
        "model_save_path": "./4_no_frozen_full.pt",
        "plot_dir": "./plots"
    }

    os.makedirs(config["plot_dir"], exist_ok=True)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

    test_set = MyDataset('split_data/test.jsonl')
    collator = GroupCollator(tokenizer=tokenizer)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'],
                             collate_fn=collator, shuffle=False, num_workers=4, pin_memory=True)

    # 模型加载
    model = SemanticComparator().to(device)
    if os.path.exists(config['model_save_path']):
        print(f"Loading checkpoint: {config['model_save_path']}")
        checkpoint = torch.load(
            config['model_save_path'],
            map_location=device,
            weights_only=False
        )
        model.load_state_dict(checkpoint['model_state'])
        model.threshold.data = checkpoint['threshold'].cpu()  # 确保阈值在CPU

    model.eval()
    test_correct = 0
    all_preds = []
    all_labels = []
    all_probs = []
    test_total = 0


    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Testing]"):
            inputs1 = [
                    [{k: v.to(device) for k, v in item.items()} for item in sample]
                    for sample in batch['groups_input1']
                ]
            inputs2 = [
                    [{k: v.to(device) for k, v in item.items()} for item in sample]
                    for sample in batch['groups_input2']
                ]

            labels = batch['labels'].to(device)

            logits, threshold = model(inputs1, inputs2)
            probas = torch.sigmoid(logits).cpu().numpy()
            preds = (probas > threshold.item()).astype(int)

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probas)
            test_total += labels.size(0)
            test_correct += (preds == labels.cpu().numpy()).sum()

    # 计算核心指标
    test_acc = test_correct / test_total
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = 2 * (precision * recall) / (precision + recall)
    cm = confusion_matrix(all_labels, all_preds)

    # 输出结果
    print("\n============= Test Evaluation =============")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # 保存混淆矩阵图
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix Visualization')
    plt.tight_layout()
    plt.savefig(os.path.join(config["plot_dir"], "confusion_matrix.png"))
    plt.close()  # 关闭当前Figure


    if len(set(all_labels)) == 2:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(config["plot_dir"], "roc_curve.png"))
        plt.close()

    # 保存分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Negative', 'Positive']))

if __name__ == "__main__":
    test()