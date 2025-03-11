import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torch.quantization as quant
from transformers import (
    BertTokenizer, BertForSequenceClassification, DistilBertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification, AutoModelForSequenceClassification
)
from datasets import load_dataset

# ----- Helper Functions -----
def count_params(model):
    return sum(p.numel() for p in model.parameters())

def model_size_in_mb(model):
    # Assuming each parameter is stored in 32-bit floats (4 bytes)
    total_bytes = sum(p.numel() for p in model.parameters()) * 4
    return total_bytes / (1024 * 1024)

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0

def load_sst2_dataset(tokenizer, split="train", limit=None):
    dataset = load_dataset("glue", "sst2", split=split)
    if limit is not None:
        dataset = dataset.select(range(limit))
    texts = dataset["sentence"]
    labels = dataset["label"]
    return texts, labels

# ----- Dataset Class -----
class HFDataset(Dataset):
    def __init__(self, tokenizer, texts, labels, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

# ----- Distillation Training Function -----
def train_quantized_distillation(teacher, student, train_loader, device, num_epochs=3, num_bits=4, temperature=2.0, alpha=0.5):
    teacher.eval()  # Teacher remains fixed.
    optimizer = optim.Adam(student.parameters(), lr=2e-5)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    scaler = GradScaler() if device.type == "cuda" else None
    
    student.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)
            
            # Teacher forward pass (no gradient)
            with torch.no_grad():
                teacher_outputs = teacher(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits
            
            if scaler is not None:
                with autocast():
                    student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
                    student_logits = student_outputs.logits
                    loss_ce = criterion_ce(student_logits, labels_batch)
                    loss_kl = criterion_kl(F.log_softmax(student_logits / temperature, dim=1),
                                           F.softmax(teacher_logits / temperature, dim=1)) * (temperature ** 2)
                    loss = alpha * loss_ce + (1 - alpha) * loss_kl
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                student_outputs = student(input_ids=input_ids, attention_mask=attention_mask)
                student_logits = student_outputs.logits
                loss_ce = criterion_ce(student_logits, labels_batch)
                loss_kl = criterion_kl(F.log_softmax(student_logits / temperature, dim=1),
                                       F.softmax(teacher_logits / temperature, dim=1)) * (temperature ** 2)
                loss = alpha * loss_ce + (1 - alpha) * loss_kl
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {epoch_loss/len(train_loader):.4f}")
    
    # Move student model to CPU and apply dynamic quantization.
    student_cpu = student.cpu()
    student_quantized = quant.quantize_dynamic(student_cpu, {nn.Linear}, dtype=torch.qint8)
    return student_quantized

# ----- Main Pipeline -----
def run_distillation_pipeline(model_pairs, device):
    results = {}
    # Load full training split for distillation training.
    train_texts, train_labels = load_sst2_dataset(tokenizer=model_pairs[0]["tokenizer"], split="train")
    train_dataset = HFDataset(model_pairs[0]["tokenizer"], train_texts, train_labels, max_length=128)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Load validation split for evaluation.
    val_texts, val_labels = load_sst2_dataset(tokenizer=model_pairs[0]["tokenizer"], split="validation")
    val_dataset = HFDataset(model_pairs[0]["tokenizer"], val_texts, val_labels, max_length=128)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    for pair in model_pairs:
        pair_name = pair["name"]
        print(f"\n=== Processing pair: {pair_name} ===")
        
        teacher = pair["teacher_model"]
        student = pair["student_model"]
        tokenizer = pair["tokenizer"]
        
        # Move models to device.
        teacher.to(device)
        student.to(device)
        
        # If multiple GPUs available, wrap models with DataParallel.
        if torch.cuda.device_count() > 1:
            teacher = torch.nn.DataParallel(teacher)
            student = torch.nn.DataParallel(student)
        
        teacher_acc = evaluate_model(teacher, val_dataloader, device)
        student_acc = evaluate_model(student, val_dataloader, device)
        teacher_params = count_params(teacher)
        student_params = count_params(student)
        teacher_size = model_size_in_mb(teacher)
        student_size = model_size_in_mb(student)
        
        print(f"Teacher ({pair['teacher_id']}): {teacher_params/1e6:.2f}M params, ~{teacher_size:.2f} MB, Val Acc: {teacher_acc*100:.2f}%")
        print(f"Student ({pair['student_id']}): {student_params/1e6:.2f}M params, ~{student_size:.2f} MB, Val Acc: {student_acc*100:.2f}%")
        
        print("Starting distillation training on full training set...")
        distilled_student = train_quantized_distillation(teacher, student, train_dataloader, device, num_epochs=3)
        
        distilled_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        distilled_acc = evaluate_model(distilled_student, distilled_dataloader, device=torch.device("cpu"))
        distilled_params = count_params(distilled_student)
        distilled_size = model_size_in_mb(distilled_student)
        
        print(f"After distillation:")
        print(f"Distilled Student ({pair['student_id']}): {distilled_params/1e6:.2f}M params, ~{distilled_size:.2f} MB, Val Acc: {distilled_acc*100:.2f}%")
        
        results[pair_name] = {
            "teacher_params": teacher_params,
            "student_params": student_params,
            "distilled_student_params": distilled_params,
            "teacher_size_mb": teacher_size,
            "student_size_mb": student_size,
            "distilled_student_size_mb": distilled_size,
            "teacher_val_acc": teacher_acc,
            "student_val_acc": student_acc,
            "distilled_student_val_acc": distilled_acc
        }
    return results

# ----- Define Two Model Pairs (dropping GPT-2 pair) -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pair 1: BERT vs DistilBERT
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
teacher_bert = BertForSequenceClassification.from_pretrained("bert-base-uncased")
student_bert = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Pair 2: RoBERTa vs DistilRoBERTa (using AutoModel for student)
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
teacher_roberta = RobertaForSequenceClassification.from_pretrained("roberta-base")
student_roberta = AutoModelForSequenceClassification.from_pretrained("distilroberta-base")

model_pairs = [
    {
        "name": "BERT vs DistilBERT",
        "teacher_id": "bert-base-uncased",
        "student_id": "distilbert-base-uncased",
        "teacher_model": teacher_bert,
        "student_model": student_bert,
        "tokenizer": bert_tokenizer
    },
    {
        "name": "RoBERTa vs DistilRoBERTa",
        "teacher_id": "roberta-base",
        "student_id": "distilroberta-base",
        "teacher_model": teacher_roberta,
        "student_model": student_roberta,
        "tokenizer": roberta_tokenizer
    }
]

results = run_distillation_pipeline(model_pairs, device)

print("\n=== Summary of Results ===")
for pair_name, info in results.items():
    print(f"{pair_name}:")
    print(f"  Teacher Val Acc: {info['teacher_val_acc']*100:.2f}%, Student Val Acc: {info['student_val_acc']*100:.2f}%, Distilled Student Val Acc: {info['distilled_student_val_acc']*100:.2f}%")
    print(f"  Teacher Params: {info['teacher_params']/1e6:.2f}M, Student Params: {info['student_params']/1e6:.2f}M, Distilled Student Params: {info['distilled_student_params']/1e6:.2f}M")
    print(f"  Teacher Size: {info['teacher_size_mb']:.2f} MB, Student Size: {info['student_size_mb']:.2f} MB, Distilled Student Size: {info['distilled_student_size_mb']:.2f} MB")
