import argparse

import rdkit
from rdkit import Chem
from props import *


def normalize_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def get_diversity(smiles_list):
    div = 0
    tot = 0
    for i in range(len(smiles_list)):
        for j in range(i+1, len(smiles_list)):
            div += 1 - similarity(smiles_list[i], smiles_list[j])
            tot += 1
    return div / tot


parser = argparse.ArgumentParser()
parser.add_argument('--reconstructed_file', type=str)
parser.add_argument('--sampled_file', type=str)
parser.add_argument('--nsample', type=int, default=10000)

args = parser.parse_args()
args.sampled_file = '/nfs/romanoff/ext01/tony/hgraph2graph/data/chembl25/sampled.txt'
args.reconstructed_file = '/nfs/romanoff/ext01/tony/hgraph2graph/data/chembl25/test_reconstructed.txt'

with open(args.sampled_file, 'r') as f:
    sampled = f.read().split('\n')[:-1]

sampled = [normalize_smiles(smiles) for smiles in sampled]
validity = len(sampled) / args.nsample
uniqueness = len(set(sampled)) / args.nsample
diversity = get_diversity(sampled)

print(f'Validity: {validity}\tUniqueness: {uniqueness}\tDiversity: {diversity}')

with open(args.reconstructed_file, 'r') as f:
    reconstructed = 0
    total = 0
    for line in f.read().split('\n')[:-1]:
        total += 1
        org, recon = line.split()
        org_smiles = normalize_smiles(org.strip())
        recon_smiles = normalize_smiles(recon.strip())
        if org_smiles == recon_smiles:
            reconstructed += 1
print(f'Reconstruction: {reconstructed / total}')
