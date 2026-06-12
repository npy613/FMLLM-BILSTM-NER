import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, LlamaForCausalLM
)

# 自定义模型类，将LLM适配为序列标注任务
class LlamaForNER(nn.Module):
    def __init__(self, model_path, hidden_dim=128, num_labels=10, max_length=512):
        super(LlamaForNER, self).__init__()
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            trust_remote_code=True,
            dtype=torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.max_length = max_length
        
        # 添加NER分类头
        hidden_size = self.model.config.hidden_size
        self.ner_classifier = nn.Linear(hidden_size, num_labels)
        # 确保分类器与主干模型数据类型一致
        self.ner_classifier = self.ner_classifier.to(self.model.dtype)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 获取最后一层隐藏状态
        hidden_states = outputs.hidden_states[-1] # [batch, seq_len, hidden_size]
        
        # 通过NER分类器得到logits
        ner_logits = self.ner_classifier(hidden_states)  # [batch, seq_len, num_labels]
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100) # 忽略labels标签为-100的部分：padding部分和指令部分
            # 只计算非padding部分的loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = ner_logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss
        
        return ner_logits

    def forward_with_attentions(self, input_ids, attention_mask):
        """返回NER logits和注意力权重，用于可视化"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )

        sequence_output = outputs.hidden_states[-1]
        ner_logits = self.ner_classifier(sequence_output)

        # outputs.attentions 是 tuple，每层一个 [batch, num_heads, seq_len, seq_len]
        return ner_logits, outputs.attentions
    
    def predict(self, input_ids, attention_mask): # no use
        """预测NER标签"""
        self.eval()
        with torch.no_grad():
            ner_logits = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(ner_logits, dim=-1)
            
            # 将预测结果转换为标签ID列表
            batch_predictions = []
            for i in range(len(predictions)):
                mask = attention_mask[i].bool()
                valid_predictions = predictions[i][mask].cpu().tolist()
                batch_predictions.append(valid_predictions)
            
            return batch_predictions