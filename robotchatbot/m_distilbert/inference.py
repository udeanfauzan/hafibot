import numpy as np
from transformers import DistilBertModel, DistilBertTokenizer
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path

class DistilBertProcessor(nn.Module):
    def __init__(self,
                model='distilbert-base-uncased',
                intent_head=15, 
                sentiment_head=4, 
                function_head=8,
                dropout=0.1,
                sequence=384):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base_dir = Path(__file__).resolve().parent.parent
        project_root = base_dir.parent
        vocab_dir = project_root / 'models' / 'distilbert-base-uncased'
        model_to_use = str(vocab_dir) if vocab_dir.exists() else model

        self.encoder = DistilBertModel.from_pretrained(model_to_use)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Sequential(
            nn.Linear(hidden, sequence),
            nn.LayerNorm(sequence),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sequence, intent_head)
        )
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden, sequence),
            nn.LayerNorm(sequence),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sequence, sentiment_head)
        )
        self.function_head = nn.Sequential(
            nn.Linear(hidden, sequence),
            nn.LayerNorm(sequence),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sequence, function_head)
        )

        self.tokenizer = DistilBertTokenizer.from_pretrained(model_to_use)

        # Prioritize DBERTModel2 checkpoint, fallback to DBERTModel
        model_path_2 = base_dir / 'models' / 'DBERTModel2' / 'Model Result.pth'
        model_path_1 = base_dir / 'models' / 'DBERTModel' / 'Model Result.pth'
        self.model_path = model_path_2 if model_path_2.exists() else model_path_1

        model_base = torch.load(self.model_path, map_location=self.device)
        self.load_state_dict(model_base[0]['best_model'])

        self.to(self.device)
        self.eval()

        # Loading label mappings
        labels_dir = Path(__file__).parent
        self.intent_labels = pd.read_csv(labels_dir / 'intent_labels.csv')
        self.intent_labels = dict(zip(self.intent_labels['id'], self.intent_labels['label']))
        self.sentiment_labels = pd.read_csv(labels_dir / 'sentiment_labels.csv')
        self.sentiment_labels = dict(zip(self.sentiment_labels['id'], self.sentiment_labels['label']))
        self.function_labels = pd.read_csv(labels_dir / 'function_labels.csv')
        self.function_labels = dict(zip(self.function_labels['id'], self.function_labels['label']))
        print(f'DistilBertProcessor loaded from {self.model_path.parent.name} on {self.device}')

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state
        bertout = hidden_state[:, 0]
        cls = self.dropout(bertout)
        intent_logits = self.intent_head(cls)
        sentiment_logits = self.sentiment_head(cls)
        function_logits = self.function_head(cls)
        return {
            'intent_logits': intent_logits,
            'sentiment_logits': sentiment_logits,
            'function_logits': function_logits
        }

    def call_inference(self, text):
        inputs = self.tokenizer(text, truncation=True, return_tensors='pt')
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

        intent_pred = torch.argmax(outputs['intent_logits'], dim=1).item()
        sentiment_pred = torch.argmax(outputs['sentiment_logits'], dim=1).item()
        function_pred = torch.argmax(outputs['function_logits'], dim=1).item()

        intent = self.intent_labels.get(intent_pred, f'intent_{intent_pred}')
        sentiment = self.sentiment_labels.get(sentiment_pred, 'neutral')
        function = self.function_labels.get(function_pred, 'chat')

        return {
            'intent': intent,
            'sentiment': sentiment,
            'function': function
        }

if __name__ == '__main__':
    bert = DistilBertProcessor()
    result = bert.call_inference('selamat pagi robot')
    print(result)
