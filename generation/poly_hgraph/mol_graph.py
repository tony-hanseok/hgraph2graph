import networkx as nx

from generation.poly_hgraph.chemutils import *
from generation.poly_hgraph.nnutils import *

add = lambda x, y: x + y if type(x) is int else (x[0] + y, x[1] + y)
add_none = lambda x, y: None if x is None else x + y


class MolGraph(object):
    BOND_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]
    MAX_POS = 20
    FRAGMENTS = None

    @staticmethod
    def load_fragments(fragments):
        fragments = [Chem.MolToSmiles(Chem.MolFromSmiles(x)) for x in fragments]
        MolGraph.FRAGMENTS = set(fragments)

    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)

        self.mol_graph = self.build_mol_graph()
        self.clusters = self.find_clusters()
        self.clusters, self.atom2clusters = self.pool_clusters()  # atom2clusters: cluster ids which have a certain atom
        self.mol_tree = self.decompose_tree()
        self.order = self.label_tree()

    def build_mol_graph(self):
        mol = self.mol
        graph = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(mol))
        for atom in mol.GetAtoms():
            graph.nodes[atom.GetIdx()]['label'] = (atom.GetSymbol(), atom.GetFormalCharge())

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = MolGraph.BOND_LIST.index(bond.GetBondType())
            graph[a1][a2]['label'] = btype
            graph[a2][a1]['label'] = btype

        return graph

    def find_clusters(self):
        mol = self.mol
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:  # special case
            return [(0,)], [[0]]

        clusters = []
        for bond in mol.GetBonds():
            if not bond.IsInRing():
                a1 = bond.GetBeginAtom().GetIdx()
                a2 = bond.GetEndAtom().GetIdx()
                clusters.append((a1, a2))

        ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
        clusters.extend(ssr)
        return clusters

    def pool_clusters(self):
        clusters = []
        visited = set()
        fragments = find_fragments(self.mol)

        for fragment_smiles, fragment_atoms in fragments:
            if fragment_smiles not in MolGraph.FRAGMENTS:  # ignore the oov
                continue
            fragment_cluster = [i for i, cls in enumerate(self.clusters) if set(cls) <= fragment_atoms]
            assert len(set(fragment_cluster) & visited) == 0
            clusters.append(list(fragment_atoms))
            visited.update(fragment_cluster)

        for i, cls in enumerate(self.clusters):
            if i not in visited:
                clusters.append(cls)

        clusters = sorted(clusters, key=lambda x: min(x))  # to ensure clusters[0] has the root node

        atom2clusters = [[] for _ in self.mol.GetAtoms()]
        for i in range(len(clusters)):
            for atom in clusters[i]:
                atom2clusters[atom].append(i)

        return clusters, atom2clusters

    def decompose_tree(self):
        clusters = self.clusters
        graph = nx.empty_graph(len(clusters))
        for atom, neighbor_clusters in enumerate(self.atom2clusters):
            if len(neighbor_clusters) <= 1:
                continue
            inter = set(self.clusters[neighbor_clusters[0]])
            for cluster_id in neighbor_clusters:
                inter = inter & set(self.clusters[cluster_id])
            assert len(inter) >= 1

            if len(neighbor_clusters) > 2 and len(inter) == 1:  # two rings + one bond has problem!
                clusters.append([atom])
                c2 = len(clusters) - 1
                graph.add_node(c2)
                for c1 in neighbor_clusters:
                    graph.add_edge(c1, c2, weight=100)
            else:
                for i, c1 in enumerate(neighbor_clusters):
                    for c2 in neighbor_clusters[i + 1:]:
                        union = set(clusters[c1]) | set(clusters[c2])
                        graph.add_edge(c1, c2, weight=len(union))

        n = len(graph.nodes)
        m = len(graph.edges)
        # assert n - m <= 1  # must be connected
        if n - m == 1:
            return graph
        else:
            return nx.maximum_spanning_tree(graph)

    def label_tree(self):
        def dfs(order, node2parent, prev_sib, curr_node, parent_node):
            node2parent[curr_node] = parent_node
            sorted_children_nodes = sorted(
                [y for y in self.mol_tree[curr_node] if y != parent_node])  # better performance with fixed order
            for idx, child_node in enumerate(sorted_children_nodes):
                self.mol_tree[curr_node][child_node]['label'] = 0
                self.mol_tree[child_node][curr_node]['label'] = idx + 1  # position encoding
                prev_sib[child_node] = sorted_children_nodes[:idx]
                prev_sib[child_node] += [curr_node, parent_node] if parent_node >= 0 else [curr_node]
                order.append((curr_node, child_node, 1))  # 1 for explore
                dfs(order, node2parent, prev_sib, child_node, curr_node)
                order.append((child_node, curr_node, 0))  # 0 for backtrack

        order, node2parent = [], {}  # node2parent: a parent node of an input node
        self.mol_tree = nx.DiGraph(self.mol_tree)
        prev_sib = [[] for _ in range(len(self.clusters))]  # prev_sib: 나를 방문하기 전 형제노드 + parent node + parent^2 node
        dfs(order, node2parent, prev_sib, 0, -1)

        order.append((0, None, 0))  # last backtrack at root

        orginal_mol = get_mol(self.smiles)
        for a in orginal_mol.GetAtoms():
            a.SetAtomMapNum(a.GetIdx() + 1)

        tree = self.mol_tree
        for i, atoms_in_cluster in enumerate(self.clusters):
            if node2parent[i] >= 0:
                atoms_in_parent_cluster = self.clusters[node2parent[i]]
                inter_atoms = set(atoms_in_cluster) & set(atoms_in_parent_cluster)
            else:
                inter_atoms = {0}
            cluster_mol, inter_label = get_inter_label(orginal_mol, atoms_in_cluster, inter_atoms, self.atom2clusters)
            # inter_label: bond를 공유하지 않는 관계일때 생성. anchor일 경우 해당 atom label=1, 아니면 atom label=0
            # cluster_mol: parent cluster와 공유하는 atom일 경우 atom label=2, parent가 아닌 다른 cluster와 공유하는 atom일 경우
            # atom_label=1, 단독으로 가지고 있는 atom일 경우 atom label=0
            tree.nodes[i]['ismiles'] = indexed_smiles = get_smiles(cluster_mol)
            tree.nodes[i]['inter_label'] = inter_label
            tree.nodes[i]['smiles'] = smiles = get_smiles(set_atommap(cluster_mol))
            if len(atoms_in_cluster) > 1:
                tree.nodes[i]['label'] = (smiles, indexed_smiles)
            else:
                tree.nodes[i]['label'] = (smiles, smiles)
            tree.nodes[i]['cluster'] = atoms_in_cluster
            tree.nodes[i]['assm_cands'] = []

            atoms_in_parent_cluster = self.clusters[node2parent[i]]
            if node2parent[i] >= 0 and len(atoms_in_parent_cluster) > 2:  # uncertainty occurs in assembly
                hist = [atom for cluster_id in prev_sib[i] for atom in self.clusters[cluster_id]]
                atoms_in_parent_cluster = self.clusters[node2parent[i]]
                tree.nodes[i]['assm_cands'] = get_assm_cands(orginal_mol, hist, inter_label, atoms_in_parent_cluster,
                                                             len(inter_atoms))

                child_order = tree[i][node2parent[i]]['label']
                diff = set(atoms_in_cluster) - set(atoms_in_parent_cluster)
                for fa_atom in inter_atoms:
                    for ch_atom in self.mol_graph[fa_atom]:
                        if ch_atom not in diff:
                            continue

                        # fa_atom: inter_atom, ch_atom:fa_atom의 neighbor 중 cluster가 가지고 있는 atom
                        label = self.mol_graph[ch_atom][fa_atom]['label']
                        if type(label) is int:  # in case one bond is assigned multiple times
                            self.mol_graph[ch_atom][fa_atom]['label'] = (label, child_order)
        return order

    @staticmethod
    def tensorize(mol_batch, vocab, avocab):
        mol_batch = [MolGraph(x) for x in mol_batch]
        tree_tensors, tree_batchG = MolGraph.tensorize_graph([x.mol_tree for x in mol_batch], vocab)
        graph_tensors, graph_batchG = MolGraph.tensorize_graph([x.mol_graph for x in mol_batch], avocab)
        tree_scope = tree_tensors[-1]
        graph_scope = graph_tensors[-1]

        max_cls_size = max([len(c) for x in mol_batch for c in x.clusters])
        cgraph = torch.zeros(len(tree_batchG) + 1, max_cls_size).int()
        for v, attr in tree_batchG.nodes(data=True):
            bid = attr['batch_id']
            offset = graph_scope[bid][0]
            tree_batchG.nodes[v]['inter_label'] = inter_label = [(x + offset, y) for x, y in attr['inter_label']]
            tree_batchG.nodes[v]['cluster'] = cls = [x + offset for x in attr['cluster']]
            tree_batchG.nodes[v]['assm_cands'] = [add(x, offset) for x in attr['assm_cands']]
            cgraph[v, :len(cls)] = torch.IntTensor(cls)

        all_orders = []
        for i, hmol in enumerate(mol_batch):
            offset = tree_scope[i][0]
            order = [(x + offset, y + offset, z) for x, y, z in hmol.order[:-1]] + [
                (hmol.order[-1][0] + offset, None, 0)]
            all_orders.append(order)

        tree_tensors = tree_tensors[:4] + (cgraph, tree_scope)
        return (tree_batchG, graph_batchG), (tree_tensors, graph_tensors), all_orders

    @staticmethod
    def tensorize_graph(graph_batch, vocab):
        fnode, fmess = [None], [(0, 0, 0, 0)]
        agraph, bgraph = [[]], [[]]
        scope = []
        edge_dict = {}
        all_G = []

        for batch_id, G in enumerate(graph_batch):
            offset = len(fnode)
            scope.append((offset, len(G)))
            G = nx.convert_node_labels_to_integers(G, first_label=offset)
            all_G.append(G)

            _fnode = [None for _ in G.nodes]
            fnode.extend(_fnode)
            _agraph = []

            try:
                for v, attr in G.nodes(data='label'):
                    # v: node id, attr: (smiles, indexed_smiles), indexed_smiles가 없어서 OOV 생길 수 있음.
                    G.nodes[v]['batch_id'] = batch_id
                    fnode[v] = vocab[attr]  # TODO: OOV
                    _agraph.append([])
            except KeyError:
                fnode = fnode[:-len(_fnode)]
                all_G = all_G[:-1]
                continue
            agraph.extend(_agraph)

            for u, v, attr in G.edges(data='label'):
                if type(attr) is tuple:
                    fmess.append((u, v, attr[0], attr[1]))
                else:
                    fmess.append((u, v, attr, 0))
                edge_dict[(u, v)] = eid = len(edge_dict) + 1
                G[u][v]['mess_idx'] = eid
                agraph[v].append(eid)
                bgraph.append([])

            for u, v in G.edges:
                eid = edge_dict[(u, v)]
                for w in G.predecessors(u):
                    if w == v:
                        continue
                    bgraph[eid].append(edge_dict[(w, u)])

        fnode[0] = fnode[1]
        fnode = torch.IntTensor(fnode)
        fmess = torch.IntTensor(fmess)
        agraph = create_pad_tensor(agraph)
        bgraph = create_pad_tensor(bgraph)

        tensor = fnode, fmess, agraph, bgraph, scope
        assert fnode.size(0) == agraph.size(0)
        assert fmess.size(0) == bgraph.size(0)
        graph = nx.union_all(all_G)
        return tensor, graph


if __name__ == "__main__":
    import sys

    test_smiles = ['CCC(NC(=O)c1scnc1C1CC1)C(=O)N1CCOCC1', 'O=C1OCCC1Sc1nnc(-c2c[nH]c3ccccc23)n1C1CC1',
                   'CCN(C)S(=O)(=O)N1CCC(Nc2cccc(OC)c2)CC1', 'CC(=O)Nc1cccc(NC(C)c2ccccn2)c1',
                   'Cc1cc(-c2nc3sc(C4CC4)nn3c2C#N)ccc1Cl', 'CCOCCCNC(=O)c1cc(OC)ccc1Br',
                   'Cc1nc(-c2ccncc2)[nH]c(=O)c1CC(=O)NC1CCCC1', 'C#CCN(CC#C)C(=O)c1cc2ccccc2cc1OC(F)F',
                   'CCOc1ccc(CN2c3ccccc3NCC2C)cc1N', 'NC(=O)C1CCC(CNc2cc(-c3ccccc3)nc3ccnn23)CC1',
                   'CC1CCc2noc(NC(=O)c3cc(=O)c4ccccc4o3)c2C1', 'c1cc(-n2cnnc2)cc(-n2cnc3ccccc32)c1',
                   'Cc1ccc(-n2nc(C)cc2NC(=O)C2CC3C=CC2C3)nn1', 'O=c1ccc(c[nH]1)C1NCCc2ccc3OCCOc3c12']

    for s in sys.stdin:  # test_smiles:
        print(s.strip("\r\n "))
        # mol = Chem.MolFromSmiles(s)
        # for a in mol.GetAtoms():
        #    a.SetAtomMapNum( a.GetIdx() )
        # print(Chem.MolToSmiles(mol))

        hmol = MolGraph(s)
        print(hmol.clusters)
        # print(list(hmol.mol_tree.edges))
        print(nx.get_node_attributes(hmol.mol_tree, 'label'))
        # print(nx.get_node_attributes(hmol.mol_tree, 'inter_label'))
        # print(nx.get_node_attributes(hmol.mol_tree, 'assm_cands'))
        # print(hmol.order)
