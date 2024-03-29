
import torch
import argparse
from torch.utils.data import Dataset, DataLoader, TensorDataset,RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn as nn
import torch.nn.functional as F
from models import GEAR
ENCODING = 'utf-8'

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='/***/bert-base-uncased', help='Random seed.')
parser.add_argument('--tiny_model_path', type=str, default='/***/bert-tiny', help='Bert max len.')
parser.add_argument('--dimension', type=int, default=128, help='bert output dimension.')
parser.add_argument('--seed', type=int, default=314, help='Random seed.')
parser.add_argument('--MAX_LEN', type=int, default=256, help='Bert max len.')
parser.add_argument('--BATCH_SIZE', type=int, default=16, help='Batch size.')
parser.add_argument('--lr_retri', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--lr_rank', type=float, default=1e-4, help='Initial learning rate.')
#parser.add_argument('--iter', type=int, default=21, help='Number of Iterations.')
parser.add_argument('--steps', type=int, default=40, help='ranker training steps')
# 145449
parser.add_argument('--epochs', type=int, default=8, help='Number of epochs to train.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument("--pool", type=str, default="att", help='Aggregating method: top, max, mean, concat, att, sum')
parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
parser.add_argument("--EVI_NUM", type=int, default=30, help='Input Evidences num.')
parser.add_argument("--TOP_K1", type=int, default=10, help='Evidences Num Selected by retriever.')
parser.add_argument("--TOP_K2", type=int, default=3, help='Evidences Num Selected by ranker.')

parser.add_argument("--train_file", type=str, default='/***/train.tsv', help='Input Training Dataset')
parser.add_argument("--dev_file", type=str, default='/***/dev.tsv', help='Validation Dataset')
parser.add_argument("--test_file", type=str, default='/***/test.tsv', help='Testing Dataset')

parser.add_argument('--patience', type=int, default=10, help='Patience')
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

class SiameseNetworkDataset(Dataset):
    def __init__(self, sample, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = sample
        self.indexes = sample["index"]
        self.labels = sample["label"]
        self.claim = sample["claim"]
        self.evidences = sample["evidences"]
        # evidences: list
    def __len__(self):
        return len(self.indexes)
      
    def tokenize(self,input_text):
        input_text = " ".join(input_text.split())
        inputs = self.tokenizer.encode_plus(
            input_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            #pad_to_max_length=True,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
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

        ids1,mask1,token_type_ids1 = self.tokenize(str(self.claim[index]))
        ids1_tensor = torch.tensor(ids1, dtype=torch.long)
        mask1_tensor = torch.tensor(mask1, dtype=torch.long)
        token_type_ids1_tensor = torch.tensor(token_type_ids1, dtype=torch.long)
        label_tensor = torch.tensor(self.labels[index], dtype=torch.long)
       
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

            claim_label.append(label_tensor)
    
        ids_claim_matrix = torch.stack(ids_claim)
        mask_claim_matrix = torch.stack(mask_claim)
        token_type_ids_claim_matrix = torch.stack(token_type_ids_claim)
        ids_evidences_matrix = torch.stack(ids_evidences)
        mask_evidences_matrix = torch.stack(mask_evidences)
        token_type_ids_evidences_matrix = torch.stack(token_type_ids_evidences)
        claim_label_matrix = torch.stack(claim_label)
       
        return {
            #'claim_index':claim_index,
            'claim_labels':claim_label_matrix,
            'ids': [ids_claim_matrix, ids_evidences_matrix],
            'mask': [mask_claim_matrix, mask_evidences_matrix],
            'token_type_ids': [token_type_ids_claim_matrix, token_type_ids_evidences_matrix],
        }

def concat(claim, evidence):
    for i in range(claim.size(0)):
        non_zero_length = torch.nonzero(claim[i]).size(0)
        combined = torch.cat((claim[i, :non_zero_length], evidence[i]))
        evidence[i] = combined[:claim.size(1)]
    return evidence

class TwinBert(nn.Module):
    def __init__(self):
        super(TwinBert, self).__init__() 
        self.model = BertModel.from_pretrained(args.tiny_model_path)
        #self.model = BertModel.from_pretrained(model_path)

    def forward_once(self, ids, mask, token_type_ids):
        output= self.model(ids, attention_mask = mask, token_type_ids = token_type_ids)
        return output[1]
    
    def forward(self, ids, mask, token_type_ids):
        output_claim = self.forward_once(ids[0],mask[0], token_type_ids[0])  
        output_evidence = self.forward_once(ids[1],mask[1], token_type_ids[1])   
        return output_claim,output_evidence

class Single_Bert(nn.Module):
    def __init__(self):
        super(Single_Bert, self).__init__() 
        self.model = BertModel.from_pretrained(args.tiny_model_path)
        #self.model = BertModel.from_pretrained(model_path)

    def forward(self, ids, mask, token_type_ids):
        output= self.model(ids, attention_mask = mask, token_type_ids = token_type_ids)
        pair_output = output[1]
        return pair_output

class Retriever(nn.Module):
    def __init__(self):
        super(Retriever, self).__init__()
        self.sel_model = TwinBert()

    def forward(self, ids, masks, token_ids, batchsize, evidences_num):  
        batch_claim_vector,batch_evidence_vector = self.sel_model(ids,masks,token_ids)  

        claim_vector = batch_claim_vector.reshape(batchsize,evidences_num,-1) 
        evidence_vector = batch_evidence_vector.reshape(batchsize,evidences_num,-1) 

        cos_tensor = F.cosine_similarity(batch_claim_vector, batch_evidence_vector)  
        cos_tensor = cos_tensor.view(batchsize,evidences_num)                        
        cos_sim = F.softmax(cos_tensor, dim=1)            
        #cos_sims = torch.FloatTensor(cos_sim)                                       
        top_cos, top_indices = torch.topk(cos_sim, args.TOP_K1, dim=1, largest=True)    

        ids_batch = [ids[0].reshape(batchsize,evidences_num,-1),ids[1].reshape(batchsize,evidences_num,-1)]   
        mask_batch = [masks[0].reshape(batchsize,evidences_num,-1),masks[1].reshape(batchsize,evidences_num,-1)]
        token_type_ids_batch = [token_ids[0].reshape(batchsize,evidences_num,-1),token_ids[1].reshape(batchsize,evidences_num,-1)]
        # ids_batch[0]:2,5,256

        #batch_ids_evidence = [[ids_batch[1][i][index] for index in indices] for i, indices in enumerate(top_indices)]  
        batch_ids_evidence = torch.stack([torch.stack([ids_batch[1][i][index] for index in top_indices[i]]) for i in range(len(top_indices))])
        batch_masks_evidence = torch.stack([torch.stack([mask_batch[1][i][index] for index in top_indices[i]]) for i in range(len(top_indices))])
        batch_token_ids_evidence = torch.stack([torch.stack([token_type_ids_batch[1][i][index] for index in top_indices[i]]) for i in range(len(top_indices))])
        # 2,3,256

        batch_ids_claim = ids_batch[0][:, :args.TOP_K1, :]
        batch_masks_claim = mask_batch[0][:, :args.TOP_K1, :]
        batch_token_ids_claim = token_type_ids_batch[0][:, :args.TOP_K1, :]
        # 2,3,256

        ids_retri = [batch_ids_claim.reshape(-1, args.MAX_LEN), batch_ids_evidence.reshape(-1, args.MAX_LEN)]
        masks_retri = [batch_masks_claim.reshape(-1, args.MAX_LEN), batch_masks_evidence.reshape(-1, args.MAX_LEN)]
        token_ids_retri = [batch_token_ids_claim.reshape(-1, args.MAX_LEN), batch_token_ids_evidence.reshape(-1, args.MAX_LEN)]
        # ids_retri[0]: 6,256

        return top_cos, ids_retri, masks_retri, token_ids_retri

class Ranker_GEAR(nn.Module):
    def __init__(self):
        super(Ranker_GEAR, self).__init__()
        self.sel_model = Single_Bert()
        #self.cla_model = GEAR(nfeat=768, nins=args.EVI_NUM, nclass=3, nlayer=1, pool='att')
        self.cla_model = GEAR(nfeat=args.dimension, nins=args.TOP_K1, nclass=3, nlayer=args.layer, pool=args.pool)

    def forward(self, ids, masks, token_ids, batchsize, evidences_num):   
        ids_pair = concat(ids[0], ids[1])
        masks_pair = concat(masks[0], masks[1])
        token_ids_pair = concat(token_ids[0], token_ids[1])    
        
        batch_pair_vector = self.sel_model(ids_pair,masks_pair,token_ids_pair) 
        batch_claim_vector = self.sel_model(ids[0],masks[0],token_ids[0]) 
        pair_vector = batch_pair_vector.reshape(batchsize,evidences_num,-1)  
        claim_vector = batch_claim_vector.reshape(batchsize,evidences_num,-1)  

        #batch_pair_relu = F.relu(torch.mm(batch_pair_vector, self.weight) + self.bias)  
        batch_pair_relu = F.relu(pair_vector).mean(dim=2)  
        pair_sims = F.softmax(batch_pair_relu, dim=1) 
        #pair_sims = torch.FloatTensor(pair_sim)   
        top_sim, top_indices = torch.topk(pair_sims, args.TOP_K2, dim=1, largest=True)
        masked_cos = torch.zeros_like(pair_sims)   
        masked_cos.scatter_(1, top_indices, top_sim)   
        masked_cos_expanded = masked_cos.unsqueeze(2).expand(-1,-1, pair_vector.size(2)) 
        masked_cos_expanded = masked_cos_expanded.to(pair_vector.device)
        evidence_sel = pair_vector * masked_cos_expanded   
        logits = self.cla_model(evidence_sel, claim_vector[:, 0])  
        return pair_sims, logits