import argparse
import os
import random
from multiprocessing import Pool

import networkx as nx
import rdkit

from generation.poly_hgraph import MolGraph, PairVocab


def process(data):
    processed = []
    for i, line in enumerate(data):

        s = line.strip("\r\n ")
        try:
            mol = MolGraph(s)
            tree = mol.mol_tree
        except AttributeError:
            continue

        G = nx.convert_node_labels_to_integers(tree)

        is_valid = True
        for _, attr in G.nodes(data='label'):
            try:
                v = vocab[attr]
            except KeyError:
                is_valid = False
                break

        if is_valid:
            processed.append(s)
    return processed


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', type=str)
    parser.add_argument('--all_file', type=str)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--ncpu', type=int, default=10)
    args = parser.parse_args()

    args.all_file = '/nfs/romanoff/ext01/tony/hgraph2graph/data/chembl25/all.txt'
    args.vocab_file = '/nfs/romanoff/ext01/tony/hgraph2graph/data/chembl25/vocab_50.txt'

    with open(args.vocab_file, 'r') as f:
        vocab = [x.strip("\r\n ").split() for x in f]
    MolGraph.load_fragments([x[0] for x in vocab if eval(x[-1])])
    vocab = PairVocab([(x, y) for x, y, _ in vocab], cuda=False)

    with open(args.all_file, 'r') as f:
        data = [line.strip("\r\n ").split()[0] for line in f if '.' not in line]

    # data = data[:1000]

    batch_size = len(data) // args.ncpu + 1
    batches = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]

    pool = Pool(args.ncpu)
    print(len(data))
    total_data = pool.map(process, batches)

    random.seed(1)
    total_data = sum(total_data, [])
    size = len(total_data)
    print(size)
    random.shuffle(total_data)

    abs_path = os.path.split(args.all_file)[0]

    with open(os.path.join(abs_path, 'all_preprocessed.txt'), 'w') as f:
        for smiles in total_data:
            f.write(f'{smiles}\n')

    with open(os.path.join(abs_path, 'train.txt'), 'w') as f:
        train_data = total_data[:int(size * 0.8)]
        for smiles in train_data:
            f.write(f'{smiles}\n')
    with open(os.path.join(abs_path, 'test.txt'), 'w') as f:
        for smiles in total_data[int(size * 0.8):int(size * 0.9)]:
            f.write(f'{smiles}\n')
    with open(os.path.join(abs_path, 'valid.txt'), 'w') as f:
        for smiles in total_data[int(size * 0.9):]:
            f.write(f'{smiles}\n')
