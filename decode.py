import argparse
import random

import rdkit
import torch
from torch.utils.data import DataLoader

from hgraph import *

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--test_file', type=str)
parser.add_argument('--vocab_file', type=str)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--model', type=str)

parser.add_argument('--num_decode', type=int, default=20)
parser.add_argument('--sample', action='store_true')
parser.add_argument('--novi', action='store_true')
parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=270)
parser.add_argument('--embed_size', type=int, default=270)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--latent_size', type=int, default=4)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=20)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)

args = parser.parse_args()

args.test_file = '/nfs/romanoff/ext01/tony/hgraph2graph/data/qed/valid.txt'
args.vocab_file = '/nfs/romanoff/ext01/tony/hgraph2graph/data/chembl25/vocab_50.txt'
args.model = '/nfs/romanoff/ext01/tony/hgraph2graph/ckpt/qed/model.11'

args.enum_root = True
args.greedy = not args.sample

args.test = [line.strip("\r\n ") for line in open(args.test_file)]
vocab = [x.strip("\r\n ").split()[:2] for x in open(args.vocab_file)]
args.vocab = PairVocab(vocab)

if args.novi:
    model = HierGNN(args).cuda()
else:
    model = HierVGNN(args).cuda()

model.load_state_dict(torch.load(args.model))
model.eval()

dataset = MolEnumRootDataset(args.test, args.vocab, args.atom_vocab)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])

torch.manual_seed(args.seed)
random.seed(args.seed)

with torch.no_grad():
    for i, batch in enumerate(loader):
        smiles = args.test[i]
        if batch is None:
            for k in range(args.num_decode):
                print(smiles, smiles)
        else:
            new_mols = model.translate(batch[1], args.num_decode, args.enum_root, args.greedy)
            for k in range(args.num_decode):
                print(smiles, new_mols[k])
