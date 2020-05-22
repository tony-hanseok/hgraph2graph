import argparse
from collections import Counter
from multiprocessing import Pool
from random import sample

from generation.poly_hgraph import *


def fragment_process(data):
    counter = Counter()
    for smiles in data:
        if '.' in smiles:
            continue

        mol = get_mol(smiles)
        try:
            fragments = find_fragments(mol)
        except AttributeError:
            continue

        for fsmiles, _ in fragments:
            counter[fsmiles] += 1
    return counter


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpu', type=int, default=5)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str, default='vocab.txt')
    args = parser.parse_args()

    args.input_file = '/home/tony/gits/hgraph2graph/data/chembl25/all.txt'
    args.output_file = '/home/tony/gits/hgraph2graph/data/chembl25/fragment_count.txt'

    with open(args.input_file, 'r') as f:
        data = f.read().split('\n')
        if data[-1] == '':
            data = data[:-1]
    data = sample(data, 500000)

    batch_size = len(data) // args.ncpu + 1
    batches = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]

    pool = Pool(args.ncpu)
    counter_list = pool.map(fragment_process, batches)
    counter = Counter()
    for cc in counter_list:
        counter += cc

    with open(args.output_file, 'w') as f:
        for fragment, count in counter.most_common():
            f.write(f'{fragment}\t{count}\n')
