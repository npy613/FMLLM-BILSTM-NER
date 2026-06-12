import pandas as pd

def read_labels(csv_file_path):
    df = pd.read_csv(csv_file_path,
                    sep=',',          
                    quotechar='"',   
                    doublequote=True, 
                    encoding='utf-8')

    tags_list = df['NE_Tags'].tolist()

    label2id = {"O": 0}  
    current_id = 1

    for tag in tags_list:
        label2id[f"B-{tag}"] = current_id
        label2id[f"I-{tag}"] = current_id + 1
        current_id += 2

    print("BIO格式的label2id:")
    print(label2id)
    return label2id