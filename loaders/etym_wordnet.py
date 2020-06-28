import random
import networkx as nx

import torch
from torch.utils import data
from torch.nn import functional as F

import corpora

class EtymWordnetDataset(data.Dataset):
	def __init__(self, nodes, edges, etym_wordnet, nneg=10):
		self.nodes = nodes
		self.edges = edges 
		self.etym_wordnet = etym_wordnet
		self.nneg = nneg 

	def len(self):
		return len(self.edges)

	def __getitem__(self, idx):
		source, target = self.edges[idx]
		examples = [
			self.nodes[source],
			self.nodes[target]
		]
		# 1 indicates neighbors,
		# 0 indicates non-neighbors
		labels = [
			1,
			1
		]

		neighbors = set(self.etym_wordnet.ancestors(source))
			| set(self.etym_wordnet.successors(source))
		for nneg_candidate in random.sample(self.nodes.keys(), self.nneg * 5):
			if nneg_candidate not in neighbors:
				examples.append(self.nodes[nneg_candidate])
				labels.append(0)
				if len(examples) >= 2 + self.nneg:
					break

		return F.one_hot(torch.tensor(examples)), labels 

def get_etym_wordnet_dataset(transitive_closure=True, nneg=10):
	etym_wordnet = corpora.get_etym_wordnet(
		relations_to_include=[corpora.EtymWordnetRelation.ETYMOLOGICAL_ORIGIN_OF], 
		format='networkx'
	)

	if transitive_closure:
		etym_wordnet = nx.transitive_closure(etym_wordnet)

	nodes = list(sorted(etym_wordnet.nodes()))
	nodes = {node:idx for (idx, node) in enumerate(nodes)}
	edges = list(etym_wordnet.edges())

	return nodes, edges, EtymWordnetDataset(nodes, edges, etym_wordnet, nneg)

