import argparse
import numpy as np

from torch import nn

from wikisent import get_wiki_sent, get_context_sent

# compare euclidean and poincare glove at different dimensionalities
# things to compare:
#		1) reconstruction
#		2) prediction (via model)

class Embeddings(nn.Module):
	def __init__(self, vocabulary, manifold, dim):
		self.manifold = manifold
		self.embeddings = nn.Embedding(len(vocabulary), dim)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train etymology embeddings')
	parser.add_argument('--manifold', dest='manifold', help='[\'euclidean\', \'poincare\']')
	parser.add_argument('--dim', dest='dim')
	args, unknown = parser.parse_known_args()

	relations = []
	

	print(len(etym_wordnet.nodes()))

	# wiki_sent = get_wiki_sent()
	# context = get_context_sent("compact", wiki_sent)
	# print(context)

