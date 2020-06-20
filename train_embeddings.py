import argparse
import numpy as np
from gensim.models.poincare import PoincareModel

from etymwordnet import get_etym_wordnet, EtymWordnetRelation

# compare euclidean and poincare glove at different dimensionalities
# things to compare:
#		1) reconstruction
#		2) prediction (via model)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train etymology embeddings')
	parser.add_argument('--type', dest='type', help='[\'poincare\']')
	args, unknown = parser.parse_known_args()#parser.parse_args()

	relations = []
	etym_wordnet = get_etym_wordnet(
		relations_to_include=[EtymWordnetRelation.IS_DERIVED_FROM])
	for edge in etym_wordnet:
		relations.append((
			'%s-%s' % (edge.source.lang, edge.source.word), 
			'%s-%s' % (edge.target.lang, edge.target.word)
		))

	model = PoincareModel(relations[0:1000], size=5, dtype=np.float32)
	model.train(epochs=1)

