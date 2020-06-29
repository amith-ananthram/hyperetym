import os
import math
import pathlib
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
BURN_IN_EPOCHS = 5
EMBEDDINGS_DIR = 'embeddings/'

def train(variant, manifold, dim, lr, batch_size, epochs):
    if torch.cuda.is_available():
        print("Using GPU...")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    nodes, edges, etym_wordnet = loaders.get_etym_wordnet_dataset(langs=['eng'])

    train_loader = data.DataLoader(
        etym_wordnet, 
        batch_size=batch_size,
        shuffle=True
    )

    if manifold == 'euclidean':
        manifold = EuclideanManifold()
    elif manifold == 'poincare':
        manifold = PoincareManifold() 
    else:
        raise Exception("Unsupported manifold: %s" % manifold)

    embeddings = Embeddings(nodes, manifold, dim).to(device)

    optimizer = RSGD(list(embeddings.parameters()), manifold, lr)

    torch.set_default_tensor_type(torch.DoubleTensor)

    embeddings.train()
    for epoch in range(epochs):
        if epoch < BURN_IN_EPOCHS:
            adjusted_lr = BURN_IN_FACTOR * lr
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
                    print('%s' % (nodes.inv[item.item()]))
                print(embedded)
                raise Exception("Saw nan.  Halting.")

            loss.backward()

            optimizer.step(lr=adjusted_lr)
            torch.cuda.empty_cache()

            if batch_id % 100 == 0:
                print('%s: EPOCH %s BATCH %s/%s, LOSS=%s' % (
                    datetime.now(), epoch, batch_id, int(len(edges) / batch_size), loss.item()))

        model_dir = os.path.join(EMBEDDINGS_DIR, variant)
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            embeddings.state_dict(),
            os.path.join(model_dir, "model_checkpoint%s.pt" % (epoch))
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train etymology embeddings')
    parser.add_argument('--variant', dest='variant', help='Name of variant')
    parser.add_argument('--manifold', dest='manifold', help='[\'euclidean\', \'poincare\']')
    parser.add_argument('--dim', dest='dim')
    parser.add_argument('--lr', dest='lr')
    parser.add_argument('--batch_size', dest='batch_size')
    parser.add_argument('--epochs', dest='epochs')
    args, unknown = parser.parse_known_args()

    train(
        variant=args.variant,
        manifold=args.manifold,
        dim=int(args.dim),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs)
    )
