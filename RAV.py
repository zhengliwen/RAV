import numpy as np
from sklearn import metrics
import transformers
import torch
import random
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset,RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch import cuda
from models import GEAR
from utils import correct_prediction
import torch.distributions as dist
import math
import os
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
parser.add_argument('--epochs', type=int, default=16, help='Number of epochs to train.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument("--pool", type=str, default="att", help='Aggregating method: top, max, mean, concat, att, sum')
parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
parser.add_argument("--EVI_NUM", type=int, default=30, help='Input Evidences num.')
parser.add_argument("--TOP_K1", type=int, default=10, help='Evidences Num Selected by retriever.')
parser.add_argument("--TOP_K2", type=int, default=3, help='Evidences Num Selected by ranker.')

parser.add_argument("--train_file", type=str, default='***/train.tsv', help='Input Training Dataset')
parser.add_argument("--dev_file", type=str, default='***/dev.tsv', help='Validation Dataset')

parser.add_argument('--patience', type=int, default=10, help='Patience')
args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained(args.model_path)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
# torch.cuda.manual seed(314) 
# torch.cuda.manual seed(3407) 

dir_path = '***/outputs/rav-%devi-%dlayer-%s-%dseed' % (args.EVI_NUM, args.layer, args.pool, args.seed)

if not os.path.exists(dir_path):
    os.mkdir(dir_path)

if os.path.exists(dir_path + '/results.txt'):
    print(dir_path + ' results exists!')
    exit(0)
else:
    print(dir_path)

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
        #claim_label_matrix = np.mat(claim_label)

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
        output_claim = self.forward_once(ids[0],mask[0], token_type_ids[0])  #claim
        output_evidence = self.forward_once(ids[1],mask[1], token_type_ids[1])   #evidence
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

        #batch_ids_evidence = [[ids_batch[1][i][index] for index in indices] for i, indices in enumerate(top_indices)]  
        batch_ids_evidence = torch.stack([torch.stack([ids_batch[1][i][index] for index in top_indices[i]]) for i in range(len(top_indices))])
        batch_masks_evidence = torch.stack([torch.stack([mask_batch[1][i][index] for index in top_indices[i]]) for i in range(len(top_indices))])
        batch_token_ids_evidence = torch.stack([torch.stack([token_type_ids_batch[1][i][index] for index in top_indices[i]]) for i in range(len(top_indices))])

        batch_ids_claim = ids_batch[0][:, :args.TOP_K1, :]
        batch_masks_claim = mask_batch[0][:, :args.TOP_K1, :]
        batch_token_ids_claim = token_type_ids_batch[0][:, :args.TOP_K1, :]

        ids_retri = [batch_ids_claim.reshape(-1, args.MAX_LEN), batch_ids_evidence.reshape(-1, args.MAX_LEN)]
        masks_retri = [batch_masks_claim.reshape(-1, args.MAX_LEN), batch_masks_evidence.reshape(-1, args.MAX_LEN)]
        token_ids_retri = [batch_token_ids_claim.reshape(-1, args.MAX_LEN), batch_token_ids_evidence.reshape(-1, args.MAX_LEN)]

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
        top_sim, top_indices = torch.topk(pair_sims, args.TOP_K2, dim=1, largest=True)
        masked_cos = torch.zeros_like(pair_sims)  
        masked_cos.scatter_(1, top_indices, top_sim)   

        masked_cos_expanded = masked_cos.unsqueeze(2).expand(-1,-1, pair_vector.size(2)) 
        masked_cos_expanded = masked_cos_expanded.to(pair_vector.device)
        evidence_sel = pair_vector * masked_cos_expanded  

        logits = self.cla_model(evidence_sel, claim_vector[:, 0])  
        #logits = self.cla_model(evidence_vector, claim_vector[:, 0])
        return pair_sims, logits

model_retri = Retriever()
model_rank = Ranker_GEAR()
device = 'cuda:0' if cuda.is_available() else 'cpu'
#device = 'cpu'
model_retri.to(device)
model_rank.to(device)

train_samples = read_samples(args.train_file)
train_set = SiameseNetworkDataset(train_samples, tokenizer, args.MAX_LEN)
train_params = {'batch_size': args.BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }
train_data_loader = DataLoader(train_set, **train_params)

dev_samples = read_samples(args.dev_file)
dev_set = SiameseNetworkDataset(dev_samples, tokenizer, args.MAX_LEN)
dev_params = {'batch_size': 10,
                'shuffle': False,
                'num_workers': 0
                }
dev_data_loader = DataLoader(dev_set, **dev_params)

best_accuracy = 0.0
patience_counter = 0
best_epoch = 0

optimizer_retri = optim.Adam(model_retri.parameters(), lr = args.lr_retri, weight_decay=0 )
optimizer_rank = optim.Adam(model_rank.parameters(), lr = args.lr_rank, weight_decay=0 )

iter = math.ceil(len(train_samples["index"])/args.BATCH_SIZE/args.steps)
#print(iter)

if __name__ == "__main__":
    for epoch in range(args.epochs):
        print('Epoch_',epoch,' Start:')
        train_running_loss = 0.0
        train_correct_pred = 0.0 
        train_retriever_loss = 0.0   
        train_tqdm_iterator = tqdm(train_data_loader)
        for index, data in enumerate(train_tqdm_iterator, 0):
        #for index, data in enumerate(train_data_loader, 0):
            ids,mask,token_type_ids = data['ids'],data['mask'],data['token_type_ids']   
            claim_labels = data['claim_labels']

            labels = claim_labels[:, 0]
            labels = labels.type(torch.LongTensor).to(device)

            ids_batch = [ids[0].view(-1, args.MAX_LEN).to(device, dtype = torch.long),ids[1].view(-1, args.MAX_LEN).to(device, dtype = torch.long)]
            mask_batch = [mask[0].view(-1, args.MAX_LEN).to(device, dtype = torch.long),mask[1].view(-1, args.MAX_LEN).to(device, dtype = torch.long)]
            token_type_ids_batch = [token_type_ids[0].view(-1, args.MAX_LEN).to(device, dtype = torch.long),token_type_ids[1].view(-1, args.MAX_LEN).to(device, dtype = torch.long)]  
                  
            iteration = 0
            for iteration in range(iter):
                # 10,256
                if iteration*args.steps <= index < (iteration+1)*args.steps:
                    model_rank.train()
                    model_retri.eval()
                    with torch.no_grad():
                        top_cos, ids_retri, masks_retri, token_ids_retri = model_retri(ids_batch, 
                                    mask_batch, token_type_ids_batch, len(labels), args.EVI_NUM)
                    
                    pair_sims, preds = model_rank(ids_retri, masks_retri, token_ids_retri, len(labels), args.TOP_K1)
                    
                    loss_label = F.nll_loss(preds, labels)

                    train_running_loss += loss_label.item()
                    train_correct_pred += correct_prediction(preds, labels)
                    train_preds_labels = preds.max(1)[1].type_as(labels)

                    optimizer_rank.zero_grad()
                    loss_label.backward()
                    optimizer_rank.step()
                description = 'Train Ranker Acc: %lf, Loss: %lf' % (train_correct_pred / (index *args.BATCH_SIZE  + len(labels)), train_running_loss / (index  + 1))
                train_tqdm_iterator.set_description(description)
                
                if index == ((iteration+1)*args.steps)-1:
                    model_rank.eval()
                    model_retri.train()
                    top_cos, ids_retri, masks_retri, token_ids_retri = model_retri(ids_batch, 
                                    mask_batch, token_type_ids_batch, len(labels), args.EVI_NUM)
                    with torch.no_grad():
                        pair_sims, preds = model_rank(ids_retri, masks_retri, token_ids_retri, len(labels), args.TOP_K1)
                    
                    sin_sims = F.log_softmax(top_cos, dim=1)  # [2,3]
                    optimizer_retri.zero_grad()
                    loss_KL = F.kl_div(sin_sims, pair_sims, reduction='batchmean')
                    train_retriever_loss += loss_KL.item()
                    loss_KL.backward()
                    optimizer_retri.step()
                    iteration += 1
            #print('Retriever Loss: %lf' % (retriever_loss / (iteration  + 1)))  

        train_loss = train_running_loss / len(train_data_loader)
        train_accuracy = train_correct_pred / len(train_data_loader.dataset)
        print("train_correct_pred:", train_correct_pred)  
        print('Train total acc: %lf, total loss: %lf\r\n' % (train_accuracy, train_loss))
        print('Train Retriever Loss: %lf' % (train_retriever_loss / iter))

        dev_running_loss = 0.0
        dev_correct_pred = 0.0
        dev_tqdm_iterator = tqdm(dev_data_loader)
        with torch.no_grad():     
            for index, data in enumerate(dev_tqdm_iterator, 0):
            #for index, data in enumerate(dev_data_loader, 0):
                model_rank.eval()
                model_retri.eval()
                ids,mask,token_type_ids = data['ids'],data['mask'],data['token_type_ids']   
                claim_labels = data['claim_labels']

                labels = claim_labels[:, 0]
                labels = labels.type(torch.LongTensor).to(device)

                ids_batch = [ids[0].view(-1, args.MAX_LEN).to(device, dtype = torch.long),ids[1].view(-1, args.MAX_LEN).to(device, dtype = torch.long)]
                mask_batch = [mask[0].view(-1, args.MAX_LEN).to(device, dtype = torch.long),mask[1].view(-1, args.MAX_LEN).to(device, dtype = torch.long)]
                token_type_ids_batch = [token_type_ids[0].view(-1, args.MAX_LEN).to(device, dtype = torch.long),token_type_ids[1].view(-1, args.MAX_LEN).to(device, dtype = torch.long)] 
                
                top_cos, ids_retri, masks_retri, token_ids_retri = model_retri(ids_batch, 
                                    mask_batch, token_type_ids_batch, len(labels), args.EVI_NUM)
                pair_sims, preds = model_rank(ids_retri, masks_retri, token_ids_retri, len(labels), args.TOP_K1)
                loss_label = F.nll_loss(preds, labels)
                dev_running_loss += loss_label.item()
                dev_correct_pred += correct_prediction(preds, labels)
                dev_preds_labels = preds.max(1)[1].type_as(labels)

        dev_loss = dev_running_loss / len(dev_data_loader)
        dev_accuracy = dev_correct_pred / len(dev_data_loader.dataset)  
        print('Validation total acc: %lf, total loss: %lf\r\n' % (dev_accuracy, dev_loss))
        #print('Validation Retriever Loss: %lf' % (dev_retriever_loss / args.iter))

        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            best_epoch = epoch
            torch.save({'epoch': epoch,
                    'model_retri':model_retri.state_dict(),
                    'model_rank':model_rank.state_dict(),
                    'best_accuracy': best_accuracy,
                    'train_losses': train_loss,
                    'dev_losses': dev_loss},
                    '%s/best.pth.tar' % dir_path)
            patience_counter = 0
        else:
            patience_counter += 1

        torch.save({'epoch': epoch,
                'model_retri':model_retri.state_dict(),
                'model_rank':model_rank.state_dict(),
                'best_accuracy': best_accuracy,
                'train_losses': train_loss,
                'dev_losses': dev_loss},
                '%s/epoch.%d.pth.tar' % (dir_path, epoch))

        if patience_counter > args.patience:
            print("Early stopping...")
            break
    print(best_epoch)
    print(best_accuracy)

    fout = open(dir_path + '/results.txt', 'w')
    fout.write('%d\t%lf\r\n' % (best_epoch, best_accuracy))
    fout.close()

             
 





                










