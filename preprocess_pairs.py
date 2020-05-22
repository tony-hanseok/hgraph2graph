import argparse
import os
import random
from multiprocessing import Pool

import networkx as nx
import rdkit

# from generation.poly_hgraph import MolGraph, PairVocab
from hgraph import MolGraph, PairVocab


def check_valid(smiles):
    try:
        mol = MolGraph(smiles)
        tree = mol.mol_tree
    except AttributeError:
        return False

    G = nx.convert_node_labels_to_integers(tree)

    is_valid = True
    for _, attr in G.nodes(data='label'):
        try:
            v = vocab[attr]
        except KeyError:
            is_valid = False
            break
    return is_valid


def process(data):
    processed = []
    for i, line in enumerate(data):

        x, y = line

        if check_valid(x.strip()) and check_valid(y.strip()):
            processed.append((x, y))
    return processed


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', type=str)
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--ncpu', type=int, default=5)
    args = parser.parse_args()

    args.train_file = '/nfs/romanoff/ext01/tony/hgraph2graph/data/logp04/train_pairs.txt'
    args.vocab_file = '/nfs/romanoff/ext01/tony/hgraph2graph/data/chembl25/vocab_50.txt'

    with open(args.vocab_file, 'r') as f:
        vocab = [x.strip("\r\n ").split() for x in f]
    # MolGraph.load_fragments([x[0] for x in vocab if eval(x[-1])])
    vocab = PairVocab([(x, y) for x, y, _ in vocab], cuda=False)


    with open(args.train_file, 'r') as f:
        data = [line.strip("\r\n ").split() for line in f if '.' not in line]

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

    abs_path = os.path.split(args.train_file)[0]

    with open(os.path.join(abs_path, 'train_pairs_preprocessed.txt'), 'w') as f:
        for smiles in total_data:
            f.write(f'{smiles[0]}\t{smiles[1]}\n')
