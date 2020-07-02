import os
import re
import glob
import math
import torch
import pickle
import argparse
import numpy as np
import networkx as nx
from colour import Color

import drawSvg as draw
from drawSvg import Drawing
from hyperbolic import euclid, poincare, util

import loaders
from train_embeddings import Embeddings 
from manifold import EuclideanManifold, PoincareManifold

from plot2 import plotPoincareDisc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot etymology embeddings')
    parser.add_argument('--min_epoch', dest='min_epoch')
    parser.add_argument('--max_epoch', dest='max_epoch')
    parser.add_argument('--start_node', dest='start_node', default='rot-root')
    parser.add_argument('--directory', dest='directory')
    parser.add_argument('--manifold', dest='manifold')
    parser.add_argument('--no_root', action='store_false', dest='include_root', default=True)
    parser.add_argument('--include_lines', action='store_true', dest='include_lines', default=False)
    parser.add_argument('--include_text', action='store_true', dest='include_text', default=False)
    args, unknown = parser.parse_known_args()

    start_node = tuple(args.start_node.split('-'))

    with open(os.path.join(args.directory, 'nodes.pkl'), 'rb') as f:
        nodes = pickle.load(f)
    with open(os.path.join(args.directory, 'edges.pkl'), 'rb') as f:
        edges = pickle.load(f)
    with open(os.path.join(args.directory, 'graph.pkl'), 'rb') as f:
        etym_wordnet = pickle.load(f)

    if args.manifold == 'euclidean':
        manifold = EuclideanManifold()
        plotfold = euclid
    else:
        manifold = PoincareManifold()
        plotfold = poincare

    torch.set_default_tensor_type(torch.DoubleTensor)
    for model_path in sorted(glob.glob(os.path.join(args.directory, 'model_checkpoint*.pt'))):
        epoch = int(re.search('model_checkpoint(\d*).pt', model_path).group(1))

        if args.min_epoch and epoch < int(args.min_epoch):
            continue

        if args.max_epoch and epoch > int(args.max_epoch):
            continue

        embeddings = Embeddings(nodes, PoincareManifold(), 2)
        embeddings.load_state_dict(torch.load(model_path, torch.device("cpu")))
        embeddings = embeddings.embeddings.weight.detach().numpy()
        embeddings = { node:embeddings[nodes[node]] for node in nodes.keys() }

        d = Drawing(2.1, 2.1, origin='center')
        d.draw(euclid.shapes.Circle(0, 0, 1), fill='silver')
        d.draw(plotfold.shapes.Point(0, 0), radius=0.01, fill='black')

        points = []
        max_level = 0
        to_traverse = [(0, ('rot', 'root')), (1, start_node)] if args.include_root else [(1, start_node)]
        while len(to_traverse) > 0:
            level, source = to_traverse.pop(0)

            max_level = max(max_level, level)
            points.append((level, plotfold.shapes.Point(*embeddings[source]), source))

            if level == 0 and source == ('rot', 'root'):
                continue

            successors = etym_wordnet.successors(source)
            for successor in successors:
                to_traverse.append((level + 1, successor))
                if args.include_lines:
                    d.draw(
                        plotfold.shapes.Line.fromPoints(
                            *embeddings[source], *embeddings[successor], segment=True
                        ), stroke_width=0.005
                    )

        colors = list(Color("red").range_to(Color("green"), max_level + 1))
        for level, point, text in points:
            d.draw(point, radius=0.01, fill=colors[level].hex)   
            if args.include_text:
                d.draw(draw.Text('%s-%s' % text, 0.01, *point))      

        d.setRenderSize(w = 800)
        d.savePng('%s-%s.png' % (args.directory, str(epoch).zfill(3)))
