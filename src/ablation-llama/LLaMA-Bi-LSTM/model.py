import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, LlamaForCausalLM
)

class LlamaForNER(nn.Module):
    def __init__(self, model_path, hidden_dim=128, num_labels=10, max_length=512):
        super(LlamaForNER, self).__init__()
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            dtype=torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.max_length = max_length
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.lstm = nn.LSTM(
            input_size=self.model.config.hidden_size,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.3)
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_labels)
       
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        sequence_output = outputs.hidden_states[-1]   
        lstm_out, _ = self.lstm(sequence_output)     
        lstm_out = self.dropout(lstm_out)
        ner_logits = self.hidden2tag(lstm_out)        

        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)           
            active_loss = attention_mask.view(-1) == 1
            active_logits = ner_logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss
        
        return ner_logits
       

    def forward_with_attentions(self, input_ids, attention_mask):
    
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )

        sequence_output = outputs.hidden_states[-1]
        lstm_out, _ = self.lstm(sequence_output)
        lstm_out = self.dropout(lstm_out)
        ner_logits = self.hidden2tag(lstm_out)

        return ner_logits, outputs.attentions
    
    def predict(self, input_ids, attention_mask):
        with torch.no_grad():
            pred = self.forward(input_ids, attention_mask)
            return pred