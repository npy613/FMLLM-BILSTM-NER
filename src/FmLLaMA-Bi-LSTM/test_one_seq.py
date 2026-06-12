import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    get_linear_schedule_with_warmup, DataCollatorForSeq2Seq
)
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import pandas as pd
import numpy as np
from datasets import Dataset
import json
from typing import List, Dict
from model import LlamaForNER
from scripts.read_labels_csv import read_labels
from scripts.read_data_txt import read_ner_file_to_lists
from seqeval.metrics import classification_report
import re
import os

def replace_numbers_with_placeholder(text, placeholder="NUM"):
    if re.search(r'\d', text):  
        return placeholder
    else:
        return text


def preprocess_function(examples, label2id, max_length, language='english'):
    texts = examples['text']
    labels_list = examples['labels']
    
    input_ids_list = []
    attention_mask_list = []
    ner_labels_list = []
    word_ids_list = [] 

    for i, text in enumerate(texts):
        word_labels = labels_list[i]
        
      
        if language == 'chinese':
            words = list(text)
        else:
            words = text.split()
        
        if len(words) != len(word_labels):
            print(f"警告: 样本 {i} 中单词和标签数量不匹配，跳过该样本")
            continue
    
        prompt = text
        text_start = prompt.find(text)
       
        encoding = tokenizer(
            prompt,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        
        
        offsets = encoding['offset_mapping'][0]
      
        word_ids = []
        id = 0
        cur_sum_len = 0
        for start, end in offsets:
            if start == end == 0:
              
                word_ids.append(-100)
            else:
                if start < text_start:
                    word_ids.append(-100)
                else:
                    token_text = prompt[start:end].strip()
                    token_text_len = len(token_text)
                    if token_text in words[id]:
                        if token_text == words[id]: 
                            word_ids.append(id)
                           
                            id += 1
                            if id >= len(words):
                                break
                        else: 
                            
                            word_ids.append(id)
                            cur_sum_len += token_text_len
                           
                            if words[id].endswith(token_text) and cur_sum_len == len(words[id]):
                                id += 1
                                cur_sum_len = 0
                                if id >= len(words):
                                    break
                    else:
                       
                        id += 1
                        if id >= len(words):
                            break 
    
        token_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx == -100:
               
                token_labels.append(-100)
            else:
               
                if 0 <= word_idx < len(word_labels):
                    if word_idx != previous_word_idx:
                        
                        token_labels.append(label2id.get(word_labels[word_idx], 0))
                    else:
                        
                        current_label = word_labels[word_idx]
                        if current_label.startswith('B-'):
                            
                            i_label = current_label.replace('B-', 'I-')
                            token_labels.append(label2id.get(i_label, 0))
                        else:
                            
                            token_labels.append(label2id.get(current_label, 0))
                else:
                    
                    token_labels.append(label2id.get('O', 0))
                
                previous_word_idx = word_idx
        
        input_ids_list.append(encoding["input_ids"][0])
        attention_mask_list.append(encoding["attention_mask"][0])
        ner_labels_list.append(torch.tensor(token_labels))
        word_ids_list.append(torch.tensor(word_ids)) 
    
    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(ner_labels_list),
        "word_ids": torch.stack(word_ids_list)
    }


def calculate_metrics(predictions, true_labels, id2label):
  
    true_labels_text = [[id2label[label] for label in true_labels]]
    pred_labels_text = [[id2label[label] for label in predictions]]
    
    report = classification_report(true_labels_text, pred_labels_text, zero_division=0, digits=4)
    return report

def evaluate_model(model, val_dataloader, device):
    model.eval()
    total_val_loss = 0
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
           
            b_input_ids = batch['input_ids'].to(device)
            b_mask = batch['attention_mask'].to(device) 
            b_labels = batch['labels'].to(device) 
            b_word_ids = batch['word_ids'] 
            
            loss = model(b_input_ids, b_mask, b_labels)
            total_val_loss += loss.item()
            
            
            ner_logits = model(b_input_ids, b_mask)  
            predictions = torch.argmax(ner_logits, dim=-1).cpu().tolist() 
            
            for i in range(len(predictions)):
                
                mask_i = b_mask[i].bool().cpu().numpy() 
                
                cur_label = b_labels[i].cpu().tolist()
                true_labels_valid = [id for id in cur_label if id != -100]
                word_ids = [id for id in b_word_ids[i].cpu().tolist() if id != -100]
                pred_labels_list = predictions[i][-len(true_labels_valid):]

                pre_word_id = None
                for idx in range(len(word_ids)):
                    if word_ids[idx] != pre_word_id:
                        pre_word_id = word_ids[idx]
                        continue
                    else:
                        pred_labels_list[idx] = -100
                        true_labels_valid[idx] = -100
                        word_ids[idx] = -100
                
                pred_labels_list = [label for label in pred_labels_list if label != -100]
                true_labels_valid = [label for label in true_labels_valid if label != -100]
                word_ids = [id for id in word_ids if id != -100]

                
                all_true_labels.extend(true_labels_valid)
                all_predictions.extend(pred_labels_list)
    
    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f"验证集总样本数: {len(all_true_labels)}")

    return avg_val_loss, all_predictions, all_true_labels

def prepare_dataset(sentences, labels, label2id, max_length, language='english'):
   
    
    processed_texts = []
    processed_labels = []
    
    for sent, labs in zip(sentences, labels):
        while len(sent) > 0:
            if sent[0] == "\"" or sent[0] == "(" or sent[0] == "-" or sent[0] == "," or sent[0] == ")" or sent[0] == "/" or sent[0] == "." or sent[0] == "'" or sent[0] == ":": 
                sent = sent[1:]
                labs = labs[1:]
            else:
                break
        if len(sent) == 0:
            continue
        text = ' '.join(sent)  
        processed_texts.append(text)      
        processed_labels.append(labs)
    
    data_dict = {
        'text': processed_texts,
        'labels': processed_labels
    }
    
    dataset = Dataset.from_dict(data_dict)
    
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, label2id, max_length, language),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def save_model(model, save_path):
    lora_save_path = save_path.replace('.pt', '_lora')
    model.model.save_pretrained(lora_save_path)
    
    ner_classifier_state = {
        'lstm_state_dict': model.lstm.state_dict(),
        'hidden2tag_state_dict': model.hidden2tag.state_dict(),
        'dropout_state_dict': model.dropout.state_dict(),
    }
    
    checkpoint = {
        'ner_classifier_state': ner_classifier_state,
        'hidden_dim': model.hidden_dim,
        'num_labels': model.num_labels,
        'max_length': model.max_length,
        'model_config': model.model.config.to_dict(),
    }
    
    torch.save(checkpoint, save_path)
    print(f"NER分类器已保存到: {save_path}")

def load_model(model, load_path, device):
    
    lora_load_path = load_path.replace('.pt', '_lora')
    if os.path.exists(lora_load_path):
      
        if not hasattr(model.model, 'peft_config'):
            model.model = PeftModel.from_pretrained(model.model, lora_load_path)
        else:
            
            model.model.load_adapter(lora_load_path, adapter_name="default")
        print("LoRA适配器加载成功")
    else:
        print("未找到LoRA适配器，使用基础模型")
    
  
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    
    if 'ner_classifier_state' in checkpoint:
        ner_state = checkpoint['ner_classifier_state']
        model.lstm.load_state_dict(ner_state['lstm_state_dict'])
        model.hidden2tag.load_state_dict(ner_state['hidden2tag_state_dict'])
        model.dropout.load_state_dict(ner_state['dropout_state_dict'])

        print("NER分类器加载成功")
    else:
        print("警告: 检查点中没有找到NER分类器状态")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")
    
    return model

def get_lora_config(inference_mode=False):
    """返回LoRA配置"""
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS, 
        inference_mode=inference_mode,
        r=16,
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"       
        ]
    )
    return lora_config


if __name__ == "__main__":
    
    tokenizer_path = "/hf_models/Llama-3.2-1B-Instruct-unsloth-bnb-4bit" 
    tags_csv_path = "/genia/label.csv"
    test_data_path = "/genia/test.txt"
    best_model_path = "/FmLLaMA-Bi-LSTM/best_weights.pt"
    language='english'
  
    label2id = read_labels(tags_csv_path)
    id2label = {v: k for k, v in label2id.items()}
   
    num_labels = len(label2id)
    
    
    max_length = 128
    hidden_dim = 128
    batch_size = 4  
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    val_sentences, val_labels = read_ner_file_to_lists(test_data_path)
  
    val_dataset = prepare_dataset(val_sentences, val_labels, label2id, max_length, language)

    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
   
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=data_collator
    )


    checkpoint = torch.load(best_model_path, weights_only=False, map_location=device)
   
    fresh_model = LlamaForNER(
        tokenizer_path, 
        hidden_dim=checkpoint.get('hidden_dim', hidden_dim),
        num_labels=checkpoint.get('num_labels', num_labels), 
        max_length=checkpoint.get('max_length', max_length)
    )
    
    lora_config = get_lora_config(inference_mode=True)
    fresh_model.model = get_peft_model(fresh_model.model, lora_config)
    
    fresh_model = load_model(fresh_model, best_model_path, device)
    fresh_model = fresh_model.to(device)

    final_val_loss, final_predictions, final_true_labels = evaluate_model(fresh_model, val_dataloader, device)

    def calculate_simple_accuracy(predictions, true_labels):
        correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0
        return accuracy, correct, total
   
    accuracy, correct, total = calculate_simple_accuracy(final_predictions, final_true_labels)
    print(f"总体准确率: {accuracy:.4f} ({correct}/{total})")

    if len(final_predictions) > 0:
        report = calculate_metrics(final_predictions, final_true_labels, id2label)
        print(report)

    print("done")

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import json

    def extract_attention_data(model, dataloader, tokenizer, id2label, device, save_json="attention_data_1.json"):
        model.eval()
        batch = next(iter(dataloader))

        b_input_ids = batch['input_ids'].to(device)
        b_mask = batch['attention_mask'].to(device)

        model.model.set_attn_implementation("eager")
        with torch.no_grad():
            ner_logits, all_attentions = model.forward_with_attentions(b_input_ids, b_mask)
        model.model.set_attn_implementation("sdpa")

        sample_idx = 0
        mask = b_mask[sample_idx].bool().cpu()

        special_ids = set(tokenizer.all_special_ids)
        all_valid = mask.nonzero(as_tuple=True)[0].tolist()
        text_pos = [pos for pos in all_valid
                    if b_input_ids[sample_idx][pos].item() not in special_ids]
        text_indices = torch.tensor(text_pos)

        token_ids = b_input_ids[sample_idx][text_indices].cpu().tolist()
        tokens = [t.replace('▁', '') if t.startswith('▁') else t
                  for t in tokenizer.convert_ids_to_tokens(token_ids)]
        preds = torch.argmax(ner_logits[sample_idx], dim=-1).cpu()
        pred_labels = [id2label.get(preds[pos].item(), 'O') for pos in text_pos]

        last_layer_attn = all_attentions[-1][sample_idx].cpu()  
        avg_attn = last_layer_attn.mean(dim=0) 
        attn_matrix = avg_attn[text_indices][:, text_indices].float().numpy()

        data_to_save = {
            'tokens': tokens,
            'pred_labels': pred_labels,
            'attention_matrix': attn_matrix.tolist()
        }

        with open(save_json, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)

        print(f"注意力数据已保存到: {save_json}")
        return save_json

    def plot_attention_heatmap(json_path, output_path="attention_heatmap.png"):
       
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tokens = data['tokens']
        pred_labels = data['pred_labels']
        attn_matrix = np.array(data['attention_matrix'])

       
        entity_types = sorted(set(lbl.split('-', 1)[1] for lbl in pred_labels if lbl != 'O'))
        paper_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488',
                        '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']
        type2color = {et: paper_colors[i % len(paper_colors)] for i, et in enumerate(entity_types)}

        def lbl_color(lbl):
            return type2color[lbl.split('-', 1)[1]] if lbl != 'O' else '#555555'

        
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.linewidth': 0.8})
        n = len(tokens)
        cell = 0.42
        fig_w = max(8, n * cell + 3)
        fig_h = max(6, n * cell + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(attn_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=0.5)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('Attention Weight', fontsize=9)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tokens, rotation=90, fontsize=8, ha='center')
        ax.set_yticklabels(tokens, fontsize=8)

        for i, (xtl, ytl) in enumerate(zip(ax.get_xticklabels(), ax.get_yticklabels())):
            c = lbl_color(pred_labels[i])
            is_entity = pred_labels[i] != 'O'
            xtl.set_color(c)
            ytl.set_color(c)
            if is_entity:
                xtl.set_fontweight('bold')
                ytl.set_fontweight('bold')

        ax.set_xlabel('Key Tokens', fontsize=10, labelpad=6)
        ax.set_ylabel('Query Tokens', fontsize=10, labelpad=6)
        ax.set_title('Attention Weights (After Softmax)\nLast Layer, Averaged Heads',
                     fontsize=11, pad=10)

        legend_handles = [mpatches.Patch(color='#555555', label='O')]
        for et, c in type2color.items():
            legend_handles.append(mpatches.Patch(color=c, label=et))
        ax.legend(handles=legend_handles, loc='upper left',
                  bbox_to_anchor=(1.12, 1.0), fontsize=8.5,
                  frameon=True, edgecolor='#cccccc')

        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.4)
        ax.tick_params(which='minor', bottom=False, left=False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"注意力热力图已保存: {output_path}")
        plt.close()

    # json_file = extract_attention_data(fresh_model, val_dataloader, tokenizer, id2label, device)
    # plot_attention_heatmap(json_file)