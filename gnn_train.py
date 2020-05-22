import argparse
import math
import os
import sys

import numpy as np
import rdkit
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from hgraph import *

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str)
parser.add_argument('--vocab_file', type=str)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--load_epoch', type=int, default=-1)

parser.add_argument('--conditional', action='store_true')
parser.add_argument('--novi', action='store_true')
parser.add_argument('--cond_size', type=int, default=4)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=270)
parser.add_argument('--embed_size', type=int, default=270)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=4)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=20)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=20.0)
parser.add_argument('--beta', type=float, default=0.3)

parser.add_argument('--epoch', type=int, default=12)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--print_iter', type=int, default=50)
parser.add_argument('--save_iter', type=int, default=-1)

args = parser.parse_args()
args.train_dir = '/nfs/romanoff/ext01/tony/hgraph2graph/data/qed/train_preprocessed'
args.vocab_file = '/nfs/romanoff/ext01/tony/hgraph2graph/data/chembl25/vocab_50.txt'
args.save_dir = '/nfs/romanoff/ext01/tony/hgraph2graph/ckpt/qed'

print(args)

vocab = [x.strip("\r\n ").split()[:2] for x in open(args.vocab_file)]
args.vocab = PairVocab(vocab)

if args.conditional:
    model = HierCondVGNN(args).cuda()
elif args.novi:
    model = HierGNN(args).cuda()
else:
    model = HierVGNN(args).cuda()

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

if args.load_epoch >= 0:
    model.load_state_dict(torch.load(args.save_dir + "/model." + str(args.load_epoch)))

print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

total_step = 0
beta = args.beta
meters = np.zeros(6)

for epoch in range(args.load_epoch + 1, args.epoch):
    dataset = DataFolder(args.train_dir, args.batch_size, shuffle=False)

    for batch in dataset:
        total_step += 1
        # print(total_step)
        batch = batch + (beta,)
        model.zero_grad()
        loss, kl_div, wacc, iacc, tacc, sacc = model(*batch)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()

        meters = meters + np.array([kl_div, loss.item(), wacc * 100, iacc * 100, tacc * 100, sacc * 100])

        if total_step % args.print_iter == 0:
            meters /= args.print_iter
            print(
                "[%d] Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (
                    total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4], meters[5],
                    param_norm(model),
                    grad_norm(model)))
            sys.stdout.flush()
            meters *= 0

        if args.save_iter >= 0 and total_step % args.save_iter == 0:
            n_iter = total_step // args.save_iter - 1
            torch.save(model.state_dict(), args.save_dir + "/model." + str(n_iter))
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])

    del dataset
    if args.save_iter == -1:
        torch.save(model.state_dict(), args.save_dir + "/model." + str(epoch))
        scheduler.step()
        print("learning rate: %.6f" % scheduler.get_lr()[0])
