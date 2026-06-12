import json
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

y_true = []
y_pred = []

VALID_LABELS = ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC", "O"]
# VALID_LABELS = ["B-CELL_LINE", "I-CELL_LINE", "B-CELL_TYPE", "I-CELL_TYPE", "B-DNA", "I-DNA", "B-PROTEIN", "I-PROTEIN", "B-RNA", "I-RNA", "O"]

with open("/LLaMA-Factory/examples/train_lora/saves/Qwen2.5-0.5B-Instruct/lora/ner-sft/predict/generated_predictions.jsonl",
          "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
    
        true_labels = item["label"].split()
        pred_labels = item["predict"].split()

        true_labels = [l if l in VALID_LABELS else "O" for l in true_labels]
        pred_labels = [l if l in VALID_LABELS else "O" for l in pred_labels]

        true_labels_len = len(true_labels)
        pred_labels_len = len(pred_labels)

        if pred_labels_len != true_labels_len:
            if pred_labels_len > true_labels_len:
                pred_labels = pred_labels[:true_labels_len]
            else:
                pred_labels = pred_labels + ["O"] * (true_labels_len - pred_labels_len)

        y_true.append(true_labels)
        y_pred.append(pred_labels)

print(classification_report(y_true, y_pred, zero_division=0, digits=4))
