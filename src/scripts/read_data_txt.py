def read_ner_file_to_lists(file_path):
    sentences = [] 
    labels = []    
    
    current_sentence = []  
    current_labels = []    
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                word = parts[0]
                label = parts[1]
                current_sentence.append(word)
                current_labels.append(label)
            elif len(parts) == 1:
                current_sentence.append(parts[0])
                current_labels.append('O')
    
    if current_sentence:
        sentences.append(current_sentence)
        labels.append(current_labels)
    
    return sentences, labels

def example_usage():
    file_path = "/home/niupeiyu/workspace/mynlp/dataset/test/test.txt"
    
    try:
        sentences, labels = read_ner_file_to_lists(file_path)
        
        print(f"成功读取 {len(sentences)} 个句子")
        print("\n前3个句子的详细信息:")
        
        for i, (sentence, label_seq) in enumerate(zip(sentences[:3], labels[:3])):
            print(f"句子 {i + 1}:")
            print(f"  文本: {' '.join(sentence)}")
            print(f"  标签: {' '.join(label_seq)}")
            print(f"  长度: {len(sentence)} 个词")
            
            entities = []
            current_entity = None
            current_entity_type = None
            current_entity_words = []
            
            for j, (word, label) in enumerate(zip(sentence, label_seq)):
                if label.startswith('B-'):
                    if current_entity is not None:
                        entities.append((current_entity_type, ' '.join(current_entity_words)))
                    current_entity_type = label[2:]
                    current_entity_words = [word]
                    current_entity = j
                elif label.startswith('I-'):
                    if current_entity is not None:
                        current_entity_words.append(word)
                else:
                    if current_entity is not None:
                        entities.append((current_entity_type, ' '.join(current_entity_words)))
                        current_entity = None
                        current_entity_type = None
                        current_entity_words = []
            
            if current_entity is not None:
                entities.append((current_entity_type, ' '.join(current_entity_words)))
            
            if entities:
                print(f"  实体: {entities}")
            print()
            
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到")
    except Exception as e:
        print(f"读取文件时出错: {e}")

def get_unique_labels(labels_list):
    unique_labels = set()
    for label_seq in labels_list:
        unique_labels.update(label_seq)
    
    return sorted(list(unique_labels))

def count_label_distribution(labels_list):
    label_count = {}
    for label_seq in labels_list:
        for label in label_seq:
            label_count[label] = label_count.get(label, 0) + 1
    
    return label_count

if __name__ == "__main__":
    example_usage()