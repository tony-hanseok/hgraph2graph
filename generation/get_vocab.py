import argparse
from multiprocessing import Pool

from rdkit import Chem

from generation.poly_hgraph import *


def process(data):
    vocab = set()
    for i, line in enumerate(data):
        if '.' in line:
            continue

        indexed_smiles = line.strip("\r\n ")
        try:
            hier_mol = MolGraph(indexed_smiles)
        except AttributeError:
            continue

        for node, attr in hier_mol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add(attr['label'])
            for _, indexed_smiles in attr['inter_label']:
                vocab.add((smiles, indexed_smiles))
    return vocab


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--min_frequency', type=int, default=50)
    parser.add_argument('--ncpu', type=int, default=5)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--fragment_count_file', type=str)
    parser.add_argument('--output_file', type=str, default='vocab.txt')
    args = parser.parse_args()

    args.input_file = '/home/tony/gits/hgraph2graph/data/chembl25/all.txt'
    args.fragment_count_file = '/home/tony/gits/hgraph2graph/data/chembl25/fragment_count.txt'
    args.output_file = f'/home/tony/gits/hgraph2graph/data/chembl25/vocab_{args.min_frequency}.txt'

    with open(args.input_file, 'r') as f:
        data = f.read().split('\n')
        if data[-1] == '':
            data = data[:-1]

    with open(args.fragment_count_file, 'r') as f:
        fragments = []
        for line in f:
            fragment, count = line.strip().split('\t')
            if int(count) < args.min_frequency:
                break

            fragments.append(fragment)
    MolGraph.load_fragments(fragments)

    batch_size = len(data) // args.ncpu + 1
    batches = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]

    pool = Pool(args.ncpu)
    vocab_list = pool.map(process, batches)
    vocab = [(x, y) for vocab in vocab_list for x, y in vocab]
    vocab = list(set(vocab))

    fragments = set(fragments)
    with open(args.output_file, 'w') as f:
        for x, y in sorted(vocab):
            cx = Chem.MolToSmiles(Chem.MolFromSmiles(x))  # dekekulize
            print(x, y, cx in fragments)
            f.write(f'{x} {y} {cx in fragments}\n')
