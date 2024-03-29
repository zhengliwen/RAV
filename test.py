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
from model_rav import read_samples, SiameseNetworkDataset, Retriever, Ranker_GEAR
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
parser.add_argument('--weight_decay', type=float, default=3e-5, help='Weight decay (L2 loss on parameters).')
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

tokenizer = BertTokenizer.from_pretrained(args.model_path)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

test_samples = read_samples(args.test_file)
test_set = SiameseNetworkDataset(test_samples, tokenizer, args.MAX_LEN)
test_params = {'batch_size': args.BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }
test_data_loader = DataLoader(test_set, **test_params)

dev_samples = read_samples(args.dev_file)
dev_set = SiameseNetworkDataset(dev_samples, tokenizer, args.MAX_LEN)
dev_params = {'batch_size': args.BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }
dev_data_loader = DataLoader(dev_set, **dev_params)

model_retri = Retriever()
model_rank = Ranker_GEAR()
device = 'cuda:0' if cuda.is_available() else 'cpu'
#device = 'cpu'
model_retri.to(device)
model_rank.to(device)

seeds = [314]

for seed in seeds:
    checkpoint = torch.load('/***/output/best.pth.tar')
    model_retri.load_state_dict(checkpoint['model_retri'])
    model_rank.load_state_dict(checkpoint['model_rank'])
    model_retri.eval()
    model_rank.eval()

    fout = open('/***/output/dev-results.tsv', 'w')
    dev_tqdm_iterator = tqdm(dev_data_loader)
    with torch.no_grad():
        for index, data in enumerate(dev_tqdm_iterator, 0):
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
            
            for i in range(preds.shape[0]):
                #fout.write('\t'.join(['%.4lf' % num for num in preds[i]]) + '\r\n')
                fout.write('\t'.join(['%.4lf' % num for num in preds[i]]) + '\n')
    fout.close()

    fout = open('/***/output/test-results.tsv', 'w')
    test_tqdm_iterator = tqdm(test_data_loader)
    with torch.no_grad():
        for index, data in enumerate(test_tqdm_iterator, 0):
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

            for i in range(preds.shape[0]):
                fout.write('\t'.join(['%.4lf' % num for num in preds[i]]) + '\r\n')
    fout.close()
