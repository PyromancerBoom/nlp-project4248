import csv
import random

def load_data(x_path, y_path):
    with open(x_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f]

    with open(y_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f]

    assert len(texts) == len(labels), "Text and label files must be the same length"
    return list(zip(texts, labels))  # [(text, label), ...]


def create_positive_entailment_pair(data, output_path):
    '''
    For each sample in the data, randomly pick another sample with the same label as its entailment.
    Pair them.
    Output the result into a csv file with 2 columns: text1, text2.
    '''
    from collections import defaultdict

    # Group samples by label
    label_to_texts = defaultdict(list)
    for text, label in data:
        label_to_texts[label].append(text)

    paired_data = []
    for text, label in data:
        candidates = label_to_texts[label]
        if len(candidates) <= 1:
            continue  # Skip if no other sample to pair with
        while True:
            pair = random.choice(candidates)
            if pair != text:
                break
        paired_data.append((text, pair))

    # Write to CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sent0', 'sent1'])  # header
        for row in paired_data:
            writer.writerow(row)


def main():
    X_PATH = "data/sarcasm_v2_train.txt"
    Y_PATH = "data/sarcasm_v2_train_label.txt"
    
    data = load_data(X_PATH, Y_PATH)
    
    OUTPUT_PATH = "data/sarcasm_v2_naiive_pair.csv"
    
    create_positive_entailment_pair(data, OUTPUT_PATH)


if __name__ == "__main__":
    main()
