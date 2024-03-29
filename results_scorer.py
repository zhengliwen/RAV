import numpy as np
#from fever.scorer import fever_score
import json
import argparse

ENCODING = 'utf-8'
parser = argparse.ArgumentParser()
parser.add_argument("--dev_file", type=str, default='/***/dev.tsv', help='Validation Dataset')
parser.add_argument("--test_file", type=str, default='/***/test.tsv', help='Testing Dataset')

args = parser.parse_args()

def read_samples(input_file):
    index=[]
    label=[]
    claim=[]
    evidences=[]
    samples = {}
    label_to_num = {'SUPPORTS': 0, 'REFUTES': 1, 'NOTENOUGHINFO': 2}
    """Read a list of `InputExample`s from an input file."""
    with open(input_file, "r", encoding='utf-8') as reader:
        for lines in reader:
            #lines = reader.readline()
            lines = lines.strip().split('\t')
            #for line in lines:
            index.append(lines[0])
            label.append(label_to_num[lines[1]])
            claim.append(lines[2])
            evidences.append(lines[3:])
        samples = {"index":index, "label":label,"claim": claim, "evidences": evidences}
    return samples

def get_predicted_label(items):
    labels = ['SUPPORTS', 'REFUTES', 'NOTENOUGHINFO']
    return labels[np.argmax(np.array(items))]

def scorer(output_file, labels):

    fin = open(output_file, 'rb')
    lines = fin.readlines()
    results = []
    pred_labels = []
    for i in range(len(lines)):
        arr = lines[i].decode(ENCODING).strip('\r\n').split('\t')
        results.append([float(arr[0]), float(arr[1]), float(arr[2])])
    fin.close()
    for result in results:
        pred_label = get_predicted_label(result)
        label_to_num = {'SUPPORTS': 0, 'REFUTES': 1, 'NOTENOUGHINFO': 2}
        pred_labels.append(label_to_num[pred_label])
    cnt = 0
    for i in range(len(labels)):
        if pred_labels[i] == labels[i]:
            cnt += 1
    score = cnt/len(labels)
    return score




dev_samples = read_samples(args.dev_file)
test_samples = read_samples(args.test_file)
dev_labels = dev_samples["label"]
test_labels = test_samples["label"]

if __name__ == '__main__':
    
    dev_score = scorer('/***/output/dev-results.tsv',dev_labels)
    test_score = scorer('/***/output/test-results.tsv',test_labels)
    
    print('Dev score:', dev_score)
    print('Test score:', test_score)
