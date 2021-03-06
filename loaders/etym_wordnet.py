import random
import networkx as nx
from bidict import bidict

import torch
from torch.utils import data
from torch.nn import functional as F

import numpy as np

import corpora

PREFIXES = [
    'a',
    'an',
    'ante',
    'anti',
    'auto',
    'circum',
    'co',
    'com',
    'con',
    'contra',
    'contro',
    'de',
    'dis',
    'en',
    'ex',
    'extra',
    'hetero',
    'homo',
    'homeo',
    'hyper',
    'il',
    'im',
    'in',
    'ir',
    'in',
    'inter',
    'intra',
    'intro',
    'macro',
    'micro',
    'mono',
    'non',
    'omni',
    'post',
    'pre',
    'pro',
    'sub',
    'sym',
    'syn',
    'tele',
    'trans',
    'tri',
    'un',
    'uni',
    'up'
]

SUFFIXES = [
    'eer',
    'er',
    'ion',
    'ity',
    'ment',
    'ness',
    'or',
    'sion',
    'ship',
    'th',
    'able',
    'ible',
    'al',
    'ant',
    'ary',
    'ful',
    'ic',
    'ious',
    'ous',
    'ive',
    'less',
    'y',
    'ed',
    'en',
    'er',
    'ing',
    'ize',
    'ise',
    'ly',
    'ward',
    'wise',
    'est',
    's',
    'es'
]

class EtymWordnetDataset(data.Dataset):
    def __init__(self, nodes, edges, etym_wordnet, nneg=10):
        self.nodes = nodes
        self.edges = edges 
        self.node_weights = np.array([
            (
                len(list(etym_wordnet.predecessors(node))) + 
                len(list(etym_wordnet.successors(node)))
            )/(2*len(edges)) for node in nodes.keys()
        ]).cumsum()
        self.etym_wordnet = etym_wordnet
        self.nneg = nneg 

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        source, target = self.edges[idx]
        examples = [
            self.nodes[source],
            self.nodes[target]
        ]

        attempts = 0
        neighbors = set(self.etym_wordnet.predecessors(source)) \
            | set(self.etym_wordnet.successors(source))
        while attempts <= self.nneg * 20:
            nneg_candidate = np.searchsorted(self.node_weights, random.random())
            if self.nodes.inv[nneg_candidate] not in neighbors:
                examples.append(self.nodes[nneg_candidate])
                if len(examples) >= 2 + self.nneg:
                    break
            attempts += 1

        if len(examples) < 2 + self.nneg:
            print("Couldn't sample enough negatives for %s, zero-padding..." % (source))
            examples.extend([0 for _ in range(len(examples) - 2 + self.nneg)])

        return torch.tensor(examples)

def get_etym_wordnet_dataset(
    langs=None, trim_ixes=True, decycle=True, transitive_closure=True, add_root=True, sub_tree_root=None, nneg=10):
    etym_wordnet = corpora.get_etym_wordnet(
        relations_to_include=[corpora.EtymWordnetRelation.ETYMOLOGICAL_ORIGIN_OF], 
        format='networkx'
    )

    if langs:
        lang_nodes = set()
        for node in etym_wordnet.nodes():
            if node[0] in langs:
                lang_nodes.add(node)

        if trim_ixes:
            filtered_nodes = set()
            for (lang, word) in lang_nodes:
                include_node = True
                for prefix in PREFIXES:
                    for suffix in SUFFIXES:
                        if word.startswith(prefix) and word.endswith(suffix):
                            base = word[len(prefix):len(word)-len(suffix)]
                            if ('eng', base) in lang_nodes:
                                include_node = False 
                                break
                for prefix in PREFIXES:
                    if word.startswith(prefix):
                        base = word[len(prefix):]
                        if ('eng', base) in lang_nodes:
                            include_node = False 
                            break 
                for suffix in SUFFIXES:
                    if word.endswith(suffix):
                        base = word[0:len(word)-len(suffix)]
                        if ('eng', base) in lang_nodes:
                            include_node = False 
                            break 
                if include_node:
                    filtered_nodes.add((lang, word))
            lang_nodes = filtered_nodes

        lang_node_ancestors = set()
        for lang_node in lang_nodes:
            lang_node_ancestors.update(nx.ancestors(etym_wordnet, lang_node))

        lang_nodes = lang_nodes | lang_node_ancestors

        # we pass it to nx.DiGraph to make an unfrozen copy
        etym_wordnet = nx.DiGraph(nx.subgraph(etym_wordnet, lang_nodes))

        has_relations = set()
        for node in etym_wordnet.nodes():
            if len(list(etym_wordnet.predecessors(node))) > 0 or len(list(etym_wordnet.successors(node))) > 0:
                has_relations.add(node)

        etym_wordnet = nx.DiGraph(nx.subgraph(etym_wordnet, has_relations))
    
    
    if decycle:
        edges = list(etym_wordnet.edges())
        ab = [a + ':'+ b +'|'+ c + ":" +d for (a,b),(c,d) in edges]
        ba = [c + ':'+ d +'|'+ a + ":" +b for (a,b),(c,d) in edges]
        val, ix = np.unique([ab, ba], return_index=True)
        tf, = np.where(np.in1d(np.arange(2*len(edges)), ix, invert=True))
        double_list = edges+edges
        dup_edges = [double_list[i] for i in tf]

        for edge in dup_edges:
            etym_wordnet.remove_edge(*edge)

        try:
            while True:
                cycle = nx.find_cycle(etym_wordnet)
                eng_root = [((a,b),(c,d)) for (a,b),(c,d) in cycle if a == 'eng']
                
                edge_remove = eng_root[0] if len(eng_root) != 0 else cycle[0]
                etym_wordnet.remove_edge(*edge_remove)
        except:
            pass

    if transitive_closure:
        etym_wordnet = nx.transitive_closure(etym_wordnet)

    if add_root:
        etym_wordnet.add_edges_from([(('rot', 'root'), i) for i in etym_wordnet.nodes \
            if not len(list(etym_wordnet.predecessors(i)))])

    if sub_tree_root:
        etym_wordnet = nx.DiGraph(nx.subgraph(
            etym_wordnet, (set(nx.descendants(etym_wordnet, sub_tree_root)) | set([sub_tree_root]))))

    nodes = list(sorted(etym_wordnet.nodes()))
    nodes = bidict({node:idx for (idx, node) in enumerate(nodes)})
    edges = list(etym_wordnet.edges())

    return nodes, edges, EtymWordnetDataset(nodes, edges, etym_wordnet, nneg)

