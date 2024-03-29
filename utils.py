import numpy as np
import torch
from sklearn import metrics
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig
ENCODING = 'utf-8'

model_path = '/***/bert-base-uncased'

def feature_pooling(features, size):
    if len(features) == 0:
        features = []
        for i in range(size):
            features.append([0.0 for _ in range(768)])

    while len(features) < size:
        features.append([0.0 for _ in range(len(features[0]))])
    return np.array(features)

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
    #return index, label, claim, evidences
    return samples

class SiameseNetworkDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = samples
        self.indexes = samples["index"]
        self.labels = samples["label"]
        self.claim = samples["claim"]
        self.evidences = samples["evidences"]
    def __len__(self):
        return len(self.indexes)
      
    def tokenize(self,input_text):
        input_text = " ".join(input_text.split())
        inputs = self.tokenizer.encode_plus(
            input_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return ids,mask,token_type_ids

    def __getitem__(self, index):
        ids_claim = []
        mask_claim =[]
        token_type_ids_claim = []
        ids_evidences = []
        mask_evidences =[]
        token_type_ids_evidences =[]
        claim_label = []
        #claim_index = [[x] * EVI_LEN for x in samples["index"]]
        #claim_label = [[x] * EVI_LEN for x in samples["label"]]
        ids1,mask1,token_type_ids1 = self.tokenize(str(self.claim[index]))
        ids1_tensor = torch.tensor(ids1, dtype=torch.long)
        mask1_tensor = torch.tensor(mask1, dtype=torch.long)
        token_type_ids1_tensor = torch.tensor(token_type_ids1, dtype=torch.long)
       
        for evidence in self.evidences[index]:
            ids2,mask2,token_type_ids2 = self.tokenize(str(evidence))
            ids2_tensor = torch.tensor(ids2, dtype=torch.long)
            mask2_tensor = torch.tensor(mask2, dtype=torch.long)
            token_type_ids2_tensor = torch.tensor(token_type_ids2, dtype=torch.long)

            ids_evidences.append(ids2_tensor)
            ids_claim.append(ids1_tensor)
            
            mask_evidences.append(mask2_tensor)
            mask_claim.append(mask1_tensor)

            token_type_ids_evidences.append(token_type_ids2_tensor)
            token_type_ids_claim.append(token_type_ids1_tensor)

            claim_label.append(self.labels[index])
        #ids2,mask2,token_type_ids2 = self.tokenize(str(self.claim[index]))
        #ids1,mask1,token_type_ids1 = self.tokenize(str(self.evidence[index]))
        ids_claim_matrix = np.mat(ids_claim)
        mask_claim_matrix = np.mat(mask_claim)
        token_type_ids_claim_matrix = np.mat(token_type_ids_claim)
        ids_evidences_matrix = np.mat(ids_evidences)
        mask_evidences_matrix = np.mat(mask_evidences)
        token_type_ids_evidences_matrix = np.mat(token_type_ids_evidences)
       
        claim_label_matrix = np.mat(claim_label)

        return {
            #'claim_index':claim_index,
            'claim_labels':claim_label_matrix,
            'ids': [ids_claim_matrix, ids_evidences_matrix],
            'mask': [mask_claim_matrix, mask_evidences_matrix],
            'token_type_ids': [token_type_ids_claim_matrix, token_type_ids_evidences_matrix],
        }

def correct_prediction(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct

class TwinBert(nn.Module):
    def __init__(self):
        super(TwinBert, self).__init__() 
        self.model = BertModel.from_pretrained(model_path)
    def forward_once(self, ids, mask, token_type_ids):
        output1= self.model(ids, attention_mask = mask, token_type_ids = token_type_ids)
        return output1[1]
    def forward(self, ids, mask, token_type_ids):
        output1 = self.forward_once(ids[0],mask[0], token_type_ids[0])  #claim
        output2 = self.forward_once(ids[1],mask[1], token_type_ids[1])   #evidence
        return output1,output2

