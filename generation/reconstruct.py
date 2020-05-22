import argparse
import os
import random
import sys

import rdkit
import torch
from rdkit import Chem
from rdkit.Chem.Draw import MolToFile
from torch.utils.data import DataLoader

from generation.poly_hgraph import *


def normalize_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--test_file', type=str)
parser.add_argument('--vocab_file', type=str)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--model', type=str)
parser.add_argument('--reconstructed_file', type=str)

parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--latent_size', type=int, default=24)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=20)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=5)
parser.add_argument('--dropout', type=float, default=0.0)

args = parser.parse_args()
args.test_file = '/nfs/romanoff/ext01/tony/hgraph2graph/data/chembl25/test.txt'
args.reconstructed_file = '/nfs/romanoff/ext01/tony/hgraph2graph/data/chembl25/test_reconstructed.txt'
args.vocab_file = '/nfs/romanoff/ext01/tony/hgraph2graph/data/chembl25/vocab_50.txt'
args.model = '/nfs/romanoff/ext01/tony/hgraph2graph/ckpt/model.12'

args.test = [line.strip("\r\n ") for line in open(args.test_file)]
vocab = [x.strip("\r\n ").split() for x in open(args.vocab_file)]
MolGraph.load_fragments([x[0] for x in vocab if eval(x[-1])])
args.vocab = PairVocab([(x, y) for x, y, _ in vocab])

model = HierVAE(args).cuda()

model.load_state_dict(torch.load(args.model))
model.eval()

dataset = MoleculeDataset(random.sample(args.test, 5000), args.vocab, args.atom_vocab, args.batch_size)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])

torch.manual_seed(args.seed)
random.seed(args.seed)

total, acc = 0, 0

abs_path = os.path.split(args.reconstructed_file)[0]
reconstructed_dir = os.path.join(abs_path, 'reconstructed')
if not os.path.exists(reconstructed_dir):
    os.makedirs(reconstructed_dir)

with open(args.reconstructed_file, 'w') as f:
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i % 10 == 0:
                sys.stdout.write('\r {} / {}'.format(i, len(loader)))
            orig_smiles = args.test[args.batch_size * i: args.batch_size * (i + 1)]
            dec_smiles = model.reconstruct(batch)
            for x, y in zip(orig_smiles, dec_smiles):
                x = normalize_smiles(x)
                y = normalize_smiles(y)
                f.write('{}\t{}\n'.format(x, y))
                try:
                    combined = '.'.join([x, y])
                    MolToFile(Chem.MolFromSmiles(combined), os.path.join(reconstructed_dir, f'{combined}.png'))
                except:
                    continue
