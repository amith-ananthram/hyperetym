import csv
from enum import Enum
from collections import Counter

DATA_PATH = 'data/etymwn.tsv'

#Counter({'rel:has_derived_form': 2264591, 'rel:is_derived_from': 2264591, 'rel:etymologically_related': 538480, 'rel:etymological_origin_of': 473415, 'rel:etymology': 473415, 'rel:variant:orthography': 16515, 'rel:derived': 2, 'rel:etymologically': 1})

class EtymWordnetRelation(Enum):
	HAS_DERIVED_FROM = 1
	IS_DERIVED_FROM = 2
	ETYMOLOGICALLY_RELATED = 3
	ETYMOLOGICAL_ORIGIN_OF = 4
	VARIANT_ORTHOGRAPHY = 5

class EtymWordnetNode:
	def __init__(self, lang, word):
		self.lang = lang
		self.word = word

class EtymWordnetEdge:
	def __init__(self, relation, source, target):
		self.relation = relation
		self.source = source
		self.target = target

def get_etym_wordnet(relations_to_include=None):
	etym_wordnet = []
	parse_errors = []
	with open(DATA_PATH, 'r') as f:
		reader = csv.reader(f, delimiter="\t")
		for row in reader:
			try:
				left_entity, relation, right_entity = row 

				left_entity_lang, left_entity = left_entity.split(':')
				right_entity_lang, right_entity = right_entity.split(':')

				if not relations_to_include or relation in relations_to_include:
					etym_wordnet.append(EtymWordnetEdge(
						relation,
						EtymWordnetNode(left_entity_lang.lower(), left_entity.lower()),
						EtymWordnetNode(right_entity_lang.lower(), right_entity.lower())
					))
			except Exception:
				parse_errors.append(row)

	if len(parse_errors) >= 0.1 * (len(parse_errors) + len(etym_wordnet)):
		raise Exception("Lots of parse errors!")

	return etym_wordnet

if __name__ == '__main__':
	etym_wordnet = get_etym_wordnet()

	counts_by_relation = Counter()
	for edge in etym_wordnet:
		counts_by_relation[edge.relation] += 1

	print(counts_by_relation)