import argparse
import numpy as np

from torch import nn
from torch.utils import data

import loaders
from rsgd import RSGD
from wikisent import get_wiki_sent, get_context_sent

# compare euclidean and poincare glove at different dimensionalities
# things to compare:
#		1) reconstruction
#		2) prediction (via model)

class Embeddings(nn.Module):
	def __init__(self, vocabulary, manifold, dim):
		self.manifold = manifold
		self.embeddings = nn.Embedding(len(vocabulary), dim)

	def forward(self, examples):
		embedded = self.embeddings(examples)
		with torch.no_grad():
			embedded = self.manifold.normalize(embedded)
		return embedded

EMBEDDINGS_DIR = 'embeddings/'


def move_to_device(batch, device):
	examples, labels = batch
	return examples.to(device), labels.to(device)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train etymology embeddings')
	parser.add_argument('--variant', dest='variant', help='Name of variant')
	parser.add_argument('--manifold', dest='manifold', help='[\'euclidean\', \'poincare\']')
	parser.add_argument('--dim', dest='dim')
	parser.add_argument('--lr', dest='lr')
	parser.add_argument('--batch_size', dest='batch_size')
	parser.add_argument('--epochs', dest='epochs')
	args, unknown = parser.parse_known_args()

	if torch.cuda.is_available():
		print("Using GPU...")
		device = torch.device("cuda:0")
	else:
		device = torch.device("cpu")

	nodes, edges, etym_wordnet = loaders.get_etym_wordnet_dataset()
	embeddings = Embeddings(nodes, None, args.dim).to(device)

	train_loader = data.DataLoader(
		etym_wordnet, 
		batch_size=int(args.batch_size),
		shuffle=True
	)

	if args.manifold == 'euclidean':
		manifold = None
	elif args.manifold == 'poincare':
		manifold = None 
	else:
		raise Exception("Unsupported manifold: %s" % args.manifold)

	optimizer = RSGD(list(embeddings.parameters()), manifold, float(args.lr))

	for epoch in range(int(args.epochs)):
		for batch_id, batch in enumerate(train_loader):
			examples, labels = move_to_device(batch, device)

			optimizer.zero_grad()
			embedded = embeddings(examples)

			targets = embedded.narrow(1, 1, embedded.size(1) - 1)
			source = embedded.narrow(1, 0, 1).expand_as(targets)
			distances = self.manifold.distance(source, targets).squeeze(-1)

			optimizer.step()
			torch.cuda.empty_cache()

		model_dir = os.path.join(EMBEDDINGS_DIR, args.variant)
		torch.save(
			embeddings.state_dict(),
			os.path.join(model_dir, "model_checkpoint%s.pt" % (epoch))
		)

