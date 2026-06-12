数据集要放在LLaMA-Factory/data中

并且在LLaMA-Factory/data/dataset_info.json中注册
如：
"conll03_train": {
    "file_name": "conll03_train.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system",
      "history": "history"
    }
  },

数据结构：
{
    "instruction": "Perform NER on the following text. Return a sequence of BIO labels for each token, separated by spaces. Use standard labels: B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC, O.",
    "input": "SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .",
    "output": "O O B-LOC O O O O B-PER O O O O",
    "system": "You are an NLP assistant proficient in named entity recognition.",
    "history": [
      [
        "A large part of Singapore 's workforce would be mobilised to ensure the meeting would run without a glitch but the average Singaporean \" would probably not be too concerned about some of the issues , \" Tan said .",
        "O O O O B-LOC O O O O O O O O O O O O O O O O O B-MISC O O O O O O O O O O O O O O B-PER O O"
      ],
      [
        "Yields on U . S . 30-year Treasury bonds were 6 . 51 percent when stock trading closed in Mexico , unchanged from Thursday .",
        "O O B-LOC O B-ORG O O O O O O O O O B-LOC O O O O O"
      ]
    ]
},

ner_train.yaml ： 放置在LLaMA-Factory/examples/train_lora
用于轻量级LLM训练 ner  自回归推理任务

ner_predict.yaml ： 放置在LLaMA-Factory/examples/train_lora
用于轻量级LLM评估 ner  自回归推理任务  主要生成含predict结果的generated_predictions.jsonl

llm_f_ner.py ： 放置在LLaMA-Factory/examples/train_lora
根据generated_predictions.jsonl计算Precision Recall F1