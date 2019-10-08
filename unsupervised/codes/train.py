import sys
import os
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from trainer import Trainer
from gnn import GNNq, GNNp
import loader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='data')
parser.add_argument('--save', type=str, default='/')
parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
parser.add_argument('--self_link_weight', type=float, default=1.0, help='Weight of self-links.')
parser.add_argument('--pre_epoch', type=int, default=200, help='Number of pre-training epochs.')
parser.add_argument('--epoch', type=int, default=200, help='Number of training epochs per iteration.')
parser.add_argument('--iter', type=int, default=10, help='Number of training iterations.')
parser.add_argument('--use_gold', type=int, default=1, help='Whether using the ground-truth label of labeled objects, 1 for using, 0 for not using.')
parser.add_argument('--tau', type=float, default=1.0, help='Annealing temperature in sampling.')
parser.add_argument('--draw', type=str, default='max', help='Method for drawing object labels, max for max-pooling, smp for sampling.')
parser.add_argument('--depth', type=int, default=1, help='Predicting neighbors within [depth] steps.')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

opt = vars(args)

net_file = opt['dataset'] + '/net.txt'
pseudo_label_file = opt['dataset'] + '/label.txt'
real_label_file = opt['dataset'] + '/label.true'
feature_file = opt['dataset'] + '/feature.txt'
pseudo_train_file = opt['dataset'] + '/train.txt'
real_train_file = opt['dataset'] + '/train.true'
dev_file = opt['dataset'] + '/dev.txt'
test_file = opt['dataset'] + '/test.txt'

vocab_node = loader.Vocab(net_file, [0, 1])
vocab_pseudo_label = loader.Vocab(pseudo_label_file, [1])
vocab_real_label = loader.Vocab(real_label_file, [1])
vocab_feature = loader.Vocab(feature_file, [1])

opt['num_node'] = len(vocab_node)
opt['num_feature'] = len(vocab_feature)
opt['num_class'] = len(vocab_pseudo_label)

graph = loader.Graph(file_name=net_file, entity=[vocab_node, 0, 1])
label_pseudo = loader.EntityLabel(file_name=pseudo_label_file, entity=[vocab_node, 0], label=[vocab_pseudo_label, 1])
label_real = loader.EntityLabel(file_name=real_label_file, entity=[vocab_node, 0], label=[vocab_real_label, 1])
feature = loader.EntityFeature(file_name=feature_file, entity=[vocab_node, 0], feature=[vocab_feature, 1])
graph.to_symmetric(opt['self_link_weight'])
feature.to_one_hot(binary=True)
adj = graph.get_sparse_adjacency(opt['cuda'])

with open(pseudo_train_file, 'r') as fi:
    idx_train_pseudo = [vocab_node.stoi[line.strip()] for line in fi]
with open(real_train_file, 'r') as fi:
    idx_train_real = [vocab_node.stoi[line.strip()] for line in fi]
with open(dev_file, 'r') as fi:
    idx_dev = [vocab_node.stoi[line.strip()] for line in fi]
with open(test_file, 'r') as fi:
    idx_test = [vocab_node.stoi[line.strip()] for line in fi]
idx_all = list(range(opt['num_node']))

inputs = torch.Tensor(feature.one_hot)
target_pseudo = torch.LongTensor(label_pseudo.itol)
target_real = torch.LongTensor(label_real.itol)
idx_train_pseudo = torch.LongTensor(idx_train_pseudo)
idx_train_real = torch.LongTensor(idx_train_real)
idx_dev = torch.LongTensor(idx_dev)
idx_test = torch.LongTensor(idx_test)
idx_all = torch.LongTensor(idx_all)
inputs_q = torch.zeros(opt['num_node'], opt['num_feature'])
target_q = torch.zeros(opt['num_node'], opt['num_class'])
inputs_p = torch.zeros(opt['num_node'], opt['num_class'])
target_p = torch.zeros(opt['num_node'], opt['num_class'])

if opt['cuda']:
    inputs = inputs.cuda()
    target_pseudo = target_pseudo.cuda()
    target_real = target_real.cuda()
    idx_train_pseudo = idx_train_pseudo.cuda()
    idx_train_real = idx_train_real.cuda()
    idx_dev = idx_dev.cuda()
    idx_test = idx_test.cuda()
    idx_all = idx_all.cuda()
    inputs_q = inputs_q.cuda()
    target_q = target_q.cuda()
    inputs_p = inputs_p.cuda()
    target_p = target_p.cuda()

gnnq = GNNq(opt, adj)
trainer_q = Trainer(opt, gnnq)

gnnp = GNNp(opt, adj)
trainer_p = Trainer(opt, gnnp)

def evaluate():
    syn = nn.Linear(opt['hidden_dim'], len(vocab_real_label))
    syn.cuda()
    gnnq.eval()
    data = trainer_q.model.predict(inputs).detach()
    lr = 0.0025
    op = optim.RMSprop(syn.parameters(), lr=lr)
    best_dev, result = 0, 0
    for k in range(100):
        logits = syn(F.dropout(data, 0.5, training=True))
        loss = F.cross_entropy(logits[idx_train_real], target_real[idx_train_real])
        op.zero_grad()
        loss.backward()
        op.step()

        logits = syn(F.dropout(data, 0.5, training=False))

        preds = torch.max(logits[idx_dev], dim=1)[1]
        correct = preds.eq(target_real[idx_dev]).double()
        accuracy_dev = correct.sum() / idx_dev.size(0)

        preds = torch.max(logits[idx_test], dim=1)[1]
        correct = preds.eq(target_real[idx_test]).double()
        accuracy_test = correct.sum() / idx_test.size(0)

        if accuracy_dev > best_dev:
            best_dev = accuracy_dev
            result = accuracy_test

    print('{:.3f}'.format(result * 100))

    return result

def init_q_data():
    inputs_q.copy_(inputs)
    temp = torch.zeros(idx_train_pseudo.size(0), target_q.size(1)).type_as(target_q)
    temp.scatter_(1, torch.unsqueeze(target_pseudo[idx_train_pseudo], 1), 1.0)
    target_q[idx_train_pseudo] = temp
    if opt['depth'] > 0:
        preds = torch.Tensor(target_q.cpu()).type_as(target_q)
        for d in range(opt['depth']):
            preds = torch.mm(adj, preds) + target_q
            for k in range(preds.size(0)):
                ones = torch.ones(preds.size(1)).cuda()
                preds[k] = torch.where(preds[k]>0, ones, preds[k])
        target_q.copy_(preds)

def update_p_data():
    preds = trainer_q.predict(inputs_q, opt['tau'])
    if opt['draw'] == 'exp':
        inputs_p.copy_(preds)
        target_p.copy_(preds)
    elif opt['draw'] == 'max':
        idx_lb = torch.max(preds, dim=-1)[1]
        inputs_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
        target_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
    elif opt['draw'] == 'smp':
        idx_lb = torch.multinomial(preds, 1).squeeze(1)
        inputs_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
        target_p.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
    if opt['use_gold'] == 1:
        temp = torch.zeros(idx_train_pseudo.size(0), target_q.size(1)).type_as(target_q)
        temp.scatter_(1, torch.unsqueeze(target_pseudo[idx_train_pseudo], 1), 1.0)
        inputs_p[idx_train_pseudo] = temp
        target_p[idx_train_pseudo] = temp

def update_q_data():
    preds = trainer_p.predict(inputs_p)
    target_q.copy_(preds)
    if opt['use_gold'] == 1:
        temp = torch.zeros(idx_train_pseudo.size(0), target_q.size(1)).type_as(target_q)
        temp.scatter_(1, torch.unsqueeze(target_pseudo[idx_train_pseudo], 1), 1.0)
        target_q[idx_train_pseudo] = temp

def pre_train(epoches):
    init_q_data()
    for epoch in range(epoches):
        loss = trainer_q.update_soft(inputs_q, target_q, idx_train_pseudo)

def train_p(epoches):
    update_p_data()
    for epoch in range(epoches):
        loss = trainer_p.update_soft(inputs_p, target_p, idx_all)

def train_q(epoches):
    update_q_data()
    for epoch in range(epoches):
        loss = trainer_q.update_soft(inputs_q, target_q, idx_all)

pre_train(opt['pre_epoch'])
for k in range(opt['iter']):
    train_p(opt['epoch'])
    train_q(opt['epoch'])

for k in range(50):
    evaluate()

if opt['save'] != '/':
    trainer_q.save(opt['save'] + '/gnnq.pt')
    trainer_p.save(opt['save'] + '/gnnp.pt')
