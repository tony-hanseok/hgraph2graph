import os
import sys
import argparse
import pickle
import random
from functools import partial
from multiprocessing import Pool

import rdkit
import torch

from generation.poly_hgraph import MolGraph, common_atom_vocab, PairVocab


def to_numpy(tensors):
    convert = lambda x: x.numpy() if type(x) is torch.Tensor else x
    a, b, c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c


def tensorize(mol_batch, vocab):
    x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab)
    return to_numpy(x)


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--vocab_file', type=str)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--ncpu', type=int, default=4)
    args = parser.parse_args()

    args.train_file = '/nfs/romanoff/ext01/tony/hgraph2graph/data/chembl25/train.txt'
    args.vocab_file = '/nfs/romanoff/ext01/tony/hgraph2graph/data/chembl25/vocab_50.txt'

    with open(args.vocab_file, 'r') as f:
        vocab = [x.strip("\r\n ").split() for x in f]
    MolGraph.load_fragments([x[0] for x in vocab if eval(x[-1])])
    vocab = PairVocab([(x, y) for x, y, _ in vocab], cuda=False)

    pool = Pool(args.ncpu)
    random.seed(1)

    abs_path = os.path.split(args.train_file)[0]
    with open(args.train_file, 'r') as f:
        train_data = [line.strip("\r\n ").split()[0] for line in f]

    random.shuffle(train_data)

    k = 10000
    # train_data = train_data[:18*k]

    for i in range(0, len(train_data), k):
        print(f'{i // k} / {len(train_data) // k}')
        _data = train_data[i:i+k]

        batches = [_data[batch_id: batch_id + args.batch_size] for batch_id in range(0, len(_data), args.batch_size)]
        func = partial(tensorize, vocab=vocab)
        all_data = pool.map(func, batches)
        num_splits = len(all_data) // 100  # len(all_data) = 100000 / 20 = 5000, num_splits = 5

        for j in range(num_splits):
            sub_data = all_data[j*100:(j+1)*100]
        #
        # le = (len(all_data) + num_splits - 1) // num_splits
        #
        # for split_id in range(num_splits):
        #     st = split_id * le
        #     sub_data = all_data[st: st + le]

            # with open(os.path.join(abs_path, 'train_preprocessed_tmp', f'tensors-{(i // k) * num_splits+j}.pkl'), 'wb') as f:
            #     pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
