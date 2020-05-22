import argparse
import os
import pickle
import random
from functools import partial
from multiprocessing import Pool

import numpy
import rdkit
import torch

from hgraph import MolGraph, common_atom_vocab, PairVocab


def to_numpy(tensors):
    convert = lambda x: x.numpy() if type(x) is torch.Tensor else x
    a, b, c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c


def tensorize(mol_batch, vocab):
    x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab)
    return to_numpy(x)


def tensorize_pair(mol_batch, vocab):
    x, y = zip(*mol_batch)
    x = MolGraph.tensorize(x, vocab, common_atom_vocab)
    y = MolGraph.tensorize(y, vocab, common_atom_vocab)
    return to_numpy(x)[:-1] + to_numpy(y)  # no need of order for x


def tensorize_cond(mol_batch, vocab):
    x, y, cond = zip(*mol_batch)
    cond = [map(int, c.split(',')) for c in cond]
    cond = numpy.array(cond)
    x = MolGraph.tensorize(x, vocab, common_atom_vocab)
    y = MolGraph.tensorize(y, vocab, common_atom_vocab)
    return to_numpy(x)[:-1] + to_numpy(y) + (cond,)  # no need of order for x


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--vocab_file', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mode', type=str, default='pair')
    parser.add_argument('--ncpu', type=int, default=20)
    args = parser.parse_args()

    args.train_file = '/nfs/romanoff/ext01/tony/hgraph2graph/data/logp04/train_pairs_preprocessed.txt'
    args.vocab_file = '/nfs/romanoff/ext01/tony/hgraph2graph/data/chembl25/vocab_50.txt'

    with open(args.vocab_file) as f:
        vocab = [x.strip("\r\n ").split()[:2] for x in f]
    vocab = PairVocab(vocab, cuda=False)

    pool = Pool(args.ncpu)
    random.seed(1)
    abs_path = os.path.split(args.train_file)[0]

    if args.mode == 'pair':
        # dataset contains molecule pairs
        with open(args.train_file) as f:
            data = [line.strip("\r\n ").split()[:2] for line in f]

        random.shuffle(data)

        print('preprocessing start')
        batches = [data[i: i + args.batch_size] for i in range(0, len(data), args.batch_size)]
        func = partial(tensorize_pair, vocab=vocab)
        all_data = pool.map(func, batches)
        print('preprocessing end')

        num_splits = max(len(all_data) // 1000, 1)

        le = (len(all_data) + num_splits - 1) // num_splits

        for split_id in range(num_splits):
            st = split_id * le
            sub_data = all_data[st: st + le]

            with open(os.path.join(abs_path, 'train_preprocessed', f'tensors-{split_id}.pkl'), 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

    elif args.mode == 'cond_pair':
        # dataset contains molecule pairs with conditions
        with open(args.train) as f:
            data = [line.strip("\r\n ").split()[:3] for line in f]

        random.shuffle(data)

        batches = [data[i: i + args.batch_size] for i in range(0, len(data), args.batch_size)]
        func = partial(tensorize_cond, vocab=vocab)
        all_data = pool.map(func, batches)
        num_splits = max(len(all_data) // 1000, 1)

        le = (len(all_data) + num_splits - 1) // num_splits

        for split_id in range(num_splits):
            st = split_id * le
            sub_data = all_data[st: st + le]

            with open(os.path.join(abs_path, 'train_preprocessed', f'tensors-{split_id}.pkl'), 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

    elif args.mode == 'single':
        # dataset contains single molecules
        with open(args.train) as f:
            data = [line.strip("\r\n ").split()[0] for line in f]

        random.shuffle(data)

        batches = [data[i: i + args.batch_size] for i in range(0, len(data), args.batch_size)]
        func = partial(tensorize, vocab=vocab)
        all_data = pool.map(func, batches)
        num_splits = len(all_data) // 1000

        le = (len(all_data) + num_splits - 1) // num_splits

        for split_id in range(num_splits):
            st = split_id * le
            sub_data = all_data[st: st + le]

            with open(os.path.join(abs_path, 'train_preprocessed', f'tensors-{split_id}.pkl'), 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
