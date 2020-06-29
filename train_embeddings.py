import os
import math
import argparse
import numpy as np
from datetime import datetime

import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F

import loaders
from rsgd import RSGD
from manifold import EuclideanManifold, PoincareManifold


class Embeddings(nn.Module):
	def __init__(self, vocabulary, manifold, dim, scale=1e-4):
		super().__init__()
		self.manifold = manifold
		self.embeddings = nn.Embedding(len(vocabulary), dim)
		self.embeddings.weight.data.uniform_(-scale, scale)

	def forward(self, examples):
		embedded = self.embeddings(examples)
		with torch.no_grad():
			embedded = self.manifold.normalize(embedded)
		return embedded

BURN_IN_FACTOR = 1/10
BURN_IN_EPOCHS = 10
EMBEDDINGS_DIR = 'embeddings/'

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

	nodes, edges, etym_wordnet = loaders.get_etym_wordnet_dataset(add_root=False)

	train_loader = data.DataLoader(
		etym_wordnet, 
		batch_size=int(args.batch_size),
		shuffle=True
	)

	if args.manifold == 'euclidean':
		manifold = EuclideanManifold()
	elif args.manifold == 'poincare':
		manifold = PoincareManifold() 
	else:
		raise Exception("Unsupported manifold: %s" % args.manifold)

	embeddings = Embeddings(nodes, manifold, int(args.dim)).to(device)

	optimizer = RSGD(list(embeddings.parameters()), manifold, float(args.lr))

	seen = set()
	embeddings.train()
	for epoch in range(int(args.epochs)):
		if epoch < BURN_IN_EPOCHS:
			adjusted_lr = BURN_IN_FACTOR * float(args.lr)
		else:
			adjusted_lr = None

		for batch_id, batch in enumerate(train_loader):
			examples = batch.to(device)

			optimizer.zero_grad()
			embedded = embeddings(examples)

			targets = embedded.narrow(1, 1, embedded.size(1) - 1)
			source = embedded.narrow(1, 0, 1).expand_as(targets)
			distances = manifold.distance(source, targets).squeeze(-1)

			# because the second example in examples is the positive
			# example, the real class label is always 0 in targets
			labels = torch.tensor([0 for _ in range(distances.shape[0])]).to(device)
			loss = F.cross_entropy(distances.neg(), labels)

			if math.isnan(loss.item()):
				print(examples)
				for item in examples[0]:
					print('%s (seen=%s)' % (nodes.inv[item.item()], item.item() in seen))
				print(embedded)
				raise Exception("Saw nan.  Halting.")

			for item in examples[0]:
				seen.add(item.item())

			loss.backward()

			optimizer.step(lr=adjusted_lr)
			torch.cuda.empty_cache()

			if batch_id % 100 == 0:
				print('%s: EPOCH %s BATCH %s/%s, LOSS=%s' % (
					datetime.now(), epoch, batch_id, len(edges), loss.item()))

		model_dir = os.path.join(EMBEDDINGS_DIR, args.variant)
		torch.save(
			embeddings.state_dict(),
			os.path.join(model_dir, "model_checkpoint%s.pt" % (epoch))
		)

